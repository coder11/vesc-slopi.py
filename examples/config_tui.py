#!/usr/bin/env python3
"""Keyboard-first TUI for editing VESC motor and app configs over TCP."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    Switch,
    Tree,
)
from textual.widgets.tree import TreeNode

from vesc_py import VescClient
from vesc_py.config_schema import CfgType, ConfigParam, ConfigSchema

ConfigKind = Literal["mcconf", "appconf"]
CONFIG_KINDS: tuple[ConfigKind, ConfigKind] = ("mcconf", "appconf")


@dataclass(frozen=True)
class ParamRef:
    kind: ConfigKind
    name: str


@dataclass(frozen=True)
class NumericValidation:
    ok: bool
    value: int | float | None
    error: str = ""


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    host, sep, port_str = endpoint.rpartition(":")
    if not sep or not host or not port_str:
        raise argparse.ArgumentTypeError("TCP endpoint must be HOST:PORT")
    try:
        port = int(port_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("TCP port must be an integer") from exc
    if port < 1 or port > 65535:
        raise argparse.ArgumentTypeError("TCP port must be in range 1..65535")
    return host, port


def _range_text(param: ConfigParam) -> str:
    if param.type == CfgType.DOUBLE:
        return f"{param.min_double:g}..{param.max_double:g}{param.suffix}"
    if param.type == CfgType.INT:
        return f"{param.min_int}..{param.max_int}{param.suffix}"
    return ""


def validate_numeric_text(param: ConfigParam, text: str) -> NumericValidation:
    """Validate a manually typed numeric config value."""

    stripped = text.strip()
    expected = (
        f"Expected {'integer' if param.type == CfgType.INT else 'number'} "
        f"between {_range_text(param)}"
    )
    if not stripped:
        return NumericValidation(False, None, expected)

    if param.type == CfgType.INT:
        if re.fullmatch(r"[+-]?\d+", stripped) is None:
            return NumericValidation(False, None, expected)
        value = int(stripped)
        if value < param.min_int or value > param.max_int:
            return NumericValidation(False, None, expected)
        return NumericValidation(True, value)

    if param.type == CfgType.DOUBLE:
        try:
            value_f = float(stripped)
        except ValueError:
            return NumericValidation(False, None, expected)
        if not math.isfinite(value_f):
            return NumericValidation(False, None, expected)
        if value_f < param.min_double or value_f > param.max_double:
            return NumericValidation(False, None, expected)
        return NumericValidation(True, value_f)

    return NumericValidation(False, None, "Selected field is not numeric")


def format_value(param: ConfigParam, value: object) -> str:
    """Format a config value for display/editing."""

    if param.type == CfgType.DOUBLE:
        number = float(value)
        decimals = max(0, param.decimals_double)
        return f"{number:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{number:.0f}"
    if param.type == CfgType.ENUM and isinstance(value, int):
        if 0 <= value < len(param.enum_names):
            return f"{param.enum_names[value]} ({value})"
    if param.type == CfgType.BOOL:
        return f"{bool(int(value))} ({int(value)})"
    if param.type == CfgType.BITFIELD:
        selected = [
            label for index, label in enumerate(param.enum_names) if int(value) & (1 << index)
        ]
        return ", ".join(selected) if selected else "None"
    return str(value)


def _select_value_is_empty(value: object) -> bool:
    return value is Select.BLANK or value is Select.NULL


def step_numeric_value(
    param: ConfigParam,
    value: object,
    direction: int,
    multiplier: int = 1,
) -> tuple[int | float, bool]:
    """Increment/decrement a numeric value and clamp to XML bounds."""

    clamped = False
    if param.type == CfgType.INT:
        next_value = int(value) + direction * max(1, param.step_int) * multiplier
        if next_value < param.min_int:
            next_value = param.min_int
            clamped = True
        if next_value > param.max_int:
            next_value = param.max_int
            clamped = True
        return next_value, clamped

    next_float = float(value) + direction * param.step_double * multiplier
    if next_float < param.min_double:
        next_float = param.min_double
        clamped = True
    if next_float > param.max_double:
        next_float = param.max_double
        clamped = True
    return next_float, clamped


class ConfirmModal(ModalScreen[bool]):
    """Simple yes/no confirmation modal."""

    CSS = """
    ConfirmModal {
        align: center middle;
    }

    #dialog {
        width: 78;
        max-width: 90%;
        height: auto;
        border: tall $primary;
        background: $surface;
        padding: 1 2;
    }

    #dialog-buttons {
        height: auto;
        align-horizontal: right;
        margin-top: 1;
    }
    """

    def __init__(self, title: str, message: str, confirm_label: str) -> None:
        super().__init__()
        self._title = title
        self._message = message
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._title)
            yield Static(self._message)
            with Horizontal(id="dialog-buttons"):
                confirm_key = self._confirm_label[:1].lower()
                yield Button(
                    f"{self._confirm_label} ({confirm_key})",
                    variant="error",
                    id="confirm",
                )
                yield Button("Cancel (c)", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#confirm", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")

    def _focus_button(self, direction: int) -> None:
        button_ids = ("confirm", "cancel")
        focused_id = getattr(self.focused, "id", None)
        index = button_ids.index(focused_id) if focused_id in button_ids else 0
        next_id = button_ids[(index + direction) % len(button_ids)]
        self.query_one(f"#{next_id}", Button).focus()

    def on_key(self, event: events.Key) -> None:
        key = event.character.lower() if event.character is not None else event.key.lower()
        if key in {"left", "right"}:
            self._focus_button(-1 if key == "left" else 1)
            event.stop()
            return
        if key == self._confirm_label[:1].lower():
            self.dismiss(True)
            event.stop()
            return
        if key in {"c", "escape"}:
            self.dismiss(False)
            event.stop()


class ApplyChangesModal(ModalScreen[bool]):
    """Review changed fields before writing them."""

    CSS = """
    ApplyChangesModal {
        align: center middle;
    }

    #apply-dialog {
        width: 110;
        max-width: 95%;
        height: 32;
        max-height: 90%;
        border: tall $primary;
        background: $surface;
        padding: 1 2;
    }

    #changes-table {
        height: 1fr;
        margin: 1 0;
    }

    #apply-buttons {
        height: auto;
        align-horizontal: right;
    }
    """

    def __init__(self, rows: list[tuple[str, str, str, str, str]]) -> None:
        super().__init__()
        self._rows = rows

    def compose(self) -> ComposeResult:
        with Vertical(id="apply-dialog"):
            yield Label(f"Apply {len(self._rows)} changed field(s)?")
            table = DataTable(id="changes-table")
            table.add_columns("Config", "Field", "Name", "Old", "New")
            for row in self._rows:
                table.add_row(*row)
            yield table
            with Horizontal(id="apply-buttons"):
                yield Button("Apply (a)", variant="success", id="apply")
                yield Button("Cancel (c)", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#apply", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "apply")

    def _focus_button(self, direction: int) -> None:
        button_ids = ("apply", "cancel")
        focused_id = getattr(self.focused, "id", None)
        index = button_ids.index(focused_id) if focused_id in button_ids else 0
        next_id = button_ids[(index + direction) % len(button_ids)]
        self.query_one(f"#{next_id}", Button).focus()

    def on_key(self, event: events.Key) -> None:
        key = event.character.lower() if event.character is not None else event.key.lower()
        if key in {"left", "right"}:
            self._focus_button(-1 if key == "left" else 1)
            event.stop()
            return
        if key == "a":
            self.dismiss(True)
            event.stop()
            return
        if key in {"c", "escape"}:
            self.dismiss(False)
            event.stop()


class HelpModal(ModalScreen[None]):
    """Keyboard help modal."""

    CSS = """
    HelpModal {
        align: center middle;
    }

    #help-dialog {
        width: 80;
        max-width: 90%;
        height: auto;
        border: tall $primary;
        background: $surface;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "\n".join(
                [
                    "Keys",
                    "",
                    "up/down: navigate, or adjust a focused numeric editor",
                    "pageup/pagedown: adjust numeric editor by 10x step",
                    "enter: edit/commit",
                    "/: search",
                    "m / a: quicknav to motor/app config",
                    "r: revert selected field",
                    "R: revert all fields",
                    "ctrl+s: review and apply changes",
                    "t: toggle dark/light theme",
                    "q or ctrl+c: quit and discard unsaved edits",
                    "escape: cancel editor/search focus",
                ]
            ),
            id="help-dialog",
        )

    def on_key(self, event: events.Key) -> None:
        if event.key in {"escape", "enter", "q"}:
            self.dismiss(None)


class ConfigTuiApp(App[None]):
    """Textual app for editing VESC configs."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
    }

    #left {
        width: 34%;
        min-width: 36;
        border: solid $primary;
        padding: 0 1;
    }

    #right {
        width: 1fr;
        border: solid $primary;
        padding: 0 1;
    }

    #tabs {
        height: auto;
        margin-bottom: 1;
    }

    #search {
        margin-bottom: 1;
    }

    #tree {
        height: 1fr;
    }

    #title {
        text-style: bold;
        margin-bottom: 1;
    }

    #editor-box {
        height: auto;
        margin: 1 0;
    }

    #enum-editor, #bool-editor, #value-input {
        display: none;
    }

    #bitfield-editor {
        height: auto;
        display: none;
    }

    #validation {
        color: $error;
        min-height: 1;
    }

    #description {
        height: 1fr;
        overflow-y: auto;
    }

    #meta, #original {
        color: $text-muted;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: $boost;
    }
    """

    BINDINGS = [
        Binding("/", "focus_search", "Search", show=True),
        Binding("ctrl+s", "apply_changes", "Apply", show=True),
        Binding("m", "switch_config('mcconf')", "Motor (quicknav)", show=True),
        Binding("a", "switch_config('appconf')", "App (quicknav)", show=True),
        Binding("r", "revert_selected", "Revert field", show=False),
        Binding("R", "revert_all", "Revert all", show=False),
        Binding("t", "toggle_theme", "Theme", show=True),
        Binding("?", "help", "Help", show=True),
        Binding("q", "request_quit", "Quit", show=True),
        Binding("ctrl+c", "request_quit", "Quit", show=False),
    ]

    def __init__(
        self,
        *,
        client: VescClient,
        endpoint: str,
        mc_schema: ConfigSchema,
        app_schema: ConfigSchema,
        mc_values: dict[str, object],
        app_values: dict[str, object],
    ) -> None:
        super().__init__()
        self.client = client
        self.endpoint = endpoint
        self.schemas: dict[ConfigKind, ConfigSchema] = {
            "mcconf": mc_schema,
            "appconf": app_schema,
        }
        self.original: dict[ConfigKind, dict[str, object]] = {
            "mcconf": dict(mc_values),
            "appconf": dict(app_values),
        }
        self.current: dict[ConfigKind, dict[str, object]] = {
            "mcconf": dict(mc_values),
            "appconf": dict(app_values),
        }
        self.active_kind: ConfigKind = "mcconf"
        self.selected: ParamRef | None = None
        self._checkboxes: list[Checkbox] = []
        self._config_nodes: dict[ConfigKind, TreeNode[object]] = {}
        self._tree_labels: dict[ParamRef, TreeNode[object]] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Static("", id="tabs")
                yield Input(placeholder="Search (/)", id="search")
                yield Tree("Configs", id="tree")
            with Vertical(id="right"):
                yield Static("Select a property", id="title")
                yield Static("", id="meta")
                with Vertical(id="editor-box"):
                    yield Input(id="value-input")
                    yield Select([], id="enum-editor")
                    yield Switch(id="bool-editor")
                    yield Vertical(id="bitfield-editor")
                    yield Static("", id="validation")
                yield Static("", id="original")
                yield Static("", id="description")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        fw = self.client.fw_version
        fw_text = f"FW {fw.major}.{fw.minor:02d} | HW {fw.hw}" if fw else "FW unknown"
        self.title = f"VESC Config TUI | {fw_text} | {self.endpoint}"
        self._set_status("Loaded motor and app configs")
        self._refresh_tabs()
        self._populate_tree()
        self.query_one("#tree", Tree).focus()

    def _schema(self, kind: ConfigKind | None = None) -> ConfigSchema:
        return self.schemas[kind or self.active_kind]

    def _values(self, kind: ConfigKind | None = None) -> dict[str, object]:
        return self.current[kind or self.active_kind]

    def _original_values(self, kind: ConfigKind | None = None) -> dict[str, object]:
        return self.original[kind or self.active_kind]

    def _param(self, ref: ParamRef) -> ConfigParam:
        return self.schemas[ref.kind].params[ref.name]

    def _dirty_refs(self) -> list[ParamRef]:
        refs: list[ParamRef] = []
        for kind in CONFIG_KINDS:
            schema = self.schemas[kind]
            for name in schema.ser_order:
                if self.current[kind].get(name) != self.original[kind].get(name):
                    refs.append(ParamRef(kind, name))
        return refs

    def _dirty_count(self) -> int:
        return len(self._dirty_refs())

    def _is_dirty(self, ref: ParamRef) -> bool:
        return self.current[ref.kind].get(ref.name) != self.original[ref.kind].get(ref.name)

    def _set_status(self, message: str) -> None:
        dirty = self._dirty_count()
        suffix = f" | {dirty} unsaved change(s)" if dirty else ""
        self.query_one("#status", Static).update(message + suffix)

    def _refresh_tabs(self) -> None:
        motor = "[Motor]" if self.active_kind == "mcconf" else " Motor "
        app = "[App]" if self.active_kind == "appconf" else " App "
        self.query_one("#tabs", Static).update(f"{motor}  {app}")

    def _matches_query(
        self,
        query: str,
        kind: ConfigKind,
        group_name: str,
        subgroup_name: str,
        name: str,
    ) -> bool:
        if not query:
            return True
        param = self.schemas[kind].params.get(name)
        if param is None:
            return False
        haystack = " ".join(
            [
                name,
                param.long_name,
                group_name,
                subgroup_name,
                param.description_text,
                param.suffix,
                " ".join(param.enum_names),
            ]
        ).lower()
        return query.lower() in haystack

    def _groups_for_schema(
        self,
        schema: ConfigSchema,
    ) -> list[tuple[str, list[tuple[str, list[str]]]]]:
        if schema.groups:
            return [
                (group.name, [(sub.name, list(sub.items)) for sub in group.subgroups])
                for group in schema.groups
            ]
        return [("All", [("Parameters", list(schema.ser_order))])]

    def _populate_tree(self) -> None:
        tree = self.query_one("#tree", Tree)
        tree.clear()
        tree.root.set_label("Configs")
        self._config_nodes.clear()
        self._tree_labels.clear()
        query = self.query_one("#search", Input).value.strip()
        expand_matches = len(query) >= 2

        config_labels: tuple[tuple[ConfigKind, str], tuple[ConfigKind, str]] = (
            ("mcconf", "Motor Config"),
            ("appconf", "App Config"),
        )
        for kind, label in config_labels:
            kind_node = tree.root.add(label, data=kind)
            self._config_nodes[kind] = kind_node
            if expand_matches or kind == self.active_kind:
                kind_node.expand()
            schema = self.schemas[kind]
            for group_name, subgroups in self._groups_for_schema(schema):
                group_node: TreeNode[object] | None = None
                for subgroup_name, items in subgroups:
                    sections: list[tuple[str | None, list[str]]] = []
                    section_label: str | None = None
                    section_items: list[str] = []
                    for item in items:
                        if item.startswith("::sep::"):
                            if section_items:
                                sections.append((section_label, section_items))
                            section_label = item.removeprefix("::sep::")
                            section_items = []
                        elif item in schema.params:
                            section_items.append(item)
                    if section_items:
                        sections.append((section_label, section_items))

                    visible_sections: list[tuple[str | None, list[str]]] = []
                    query_lower = query.lower()
                    for section_label, section_items in sections:
                        section_matches = bool(
                            query_lower
                            and section_label is not None
                            and query_lower in section_label.lower()
                        )
                        visible_items = [
                            item
                            for item in section_items
                            if section_matches
                            or self._matches_query(query, kind, group_name, subgroup_name, item)
                        ]
                        if visible_items:
                            visible_sections.append((section_label, visible_items))

                    if not visible_sections:
                        continue
                    if group_node is None:
                        group_node = kind_node.add(group_name)
                        if expand_matches or kind == self.active_kind:
                            group_node.expand()
                    subgroup_node = group_node.add(subgroup_name)
                    if expand_matches or kind == self.active_kind:
                        subgroup_node.expand()
                    for section_label, visible_items in visible_sections:
                        parent_node = subgroup_node
                        if section_label is not None:
                            parent_node = subgroup_node.add(section_label)
                            if expand_matches or kind == self.active_kind:
                                parent_node.expand()
                        for item in visible_items:
                            ref = ParamRef(kind, item)
                            node = parent_node.add_leaf(self._node_label(ref), data=ref)
                            self._tree_labels[ref] = node

        tree.root.expand()

    def _node_label(self, ref: ParamRef) -> str:
        param = self._param(ref)
        label = param.long_name or ref.name
        return f"{label} *" if self._is_dirty(ref) else label

    def on_tree_node_selected(self, event: Tree.NodeSelected[object]) -> None:
        data = event.node.data
        if data == "mcconf" or data == "appconf":
            self.action_switch_config(data)
            return
        if isinstance(data, ParamRef):
            self.selected = data
            self._show_selected()
            self._focus_editor_for_selected()

    def _hide_editors(self) -> None:
        for selector in ("#value-input", "#enum-editor", "#bool-editor", "#bitfield-editor"):
            self.query_one(selector).styles.display = "none"
        self.query_one("#validation", Static).update("")

    def _show_selected(self) -> None:
        if self.selected is None:
            return
        ref = self.selected
        param = self._param(ref)
        value = self.current[ref.kind].get(ref.name, "")
        self._hide_editors()
        self.query_one("#title", Static).update(param.long_name or ref.name)
        self.query_one("#meta", Static).update(
            f"{ref.kind} | {ref.name} | {param.type.name}"
            + (f" | range {_range_text(param)}" if _range_text(param) else "")
        )
        original = self.original[ref.kind].get(ref.name, "")
        if self._is_dirty(ref):
            self.query_one("#original", Static).update(
                f"Original: {format_value(param, original)} | Current: {format_value(param, value)}"
            )
        else:
            self.query_one("#original", Static).update(f"Current: {format_value(param, value)}")
        self.query_one("#description", Static).update(param.description_text or "No description.")

        if param.type in (CfgType.DOUBLE, CfgType.INT, CfgType.QSTRING):
            editor = self.query_one("#value-input", Input)
            editor.value = (
                format_value(param, value)
                if param.type != CfgType.QSTRING
                else str(value)
            )
            editor.placeholder = _range_text(param) or "Value"
            editor.styles.display = "block"
        elif param.type == CfgType.ENUM:
            editor = self.query_one("#enum-editor", Select)
            options = [(label, index) for index, label in enumerate(param.enum_names)]
            with editor.prevent(Select.Changed):
                editor.set_options(options)
                editor.value = int(value)
            editor.styles.display = "block"
        elif param.type == CfgType.BOOL:
            editor = self.query_one("#bool-editor", Switch)
            editor.value = bool(int(value))
            editor.styles.display = "block"
        elif param.type == CfgType.BITFIELD:
            container = self.query_one("#bitfield-editor", Vertical)
            container.remove_children()
            self._checkboxes = []
            int_value = int(value)
            for index, label in enumerate(param.enum_names):
                checkbox = Checkbox(label, value=bool(int_value & (1 << index)), id=f"bit-{index}")
                self._checkboxes.append(checkbox)
                container.mount(checkbox)
            container.styles.display = "block"
        else:
            self.query_one("#validation", Static).update("Unsupported config type; read-only")

    def _focus_editor_for_selected(self) -> None:
        if self.selected is None:
            return
        param = self._param(self.selected)
        if param.type in (CfgType.DOUBLE, CfgType.INT, CfgType.QSTRING):
            self.query_one("#value-input", Input).focus()
        elif param.type == CfgType.ENUM:
            self.query_one("#enum-editor", Select).focus()
        elif param.type == CfgType.BOOL:
            self.query_one("#bool-editor", Switch).focus()
        elif param.type == CfgType.BITFIELD and self._checkboxes:
            self._checkboxes[0].focus()

    def _commit_value(self, ref: ParamRef, value: object) -> None:
        self.current[ref.kind][ref.name] = value
        node = self._tree_labels.get(ref)
        if node is not None:
            node.set_label(self._node_label(ref))
        self._show_selected()
        self._set_status("Edited value")

    def _commit_text_editor(self) -> bool:
        if self.selected is None:
            return False
        ref = self.selected
        param = self._param(ref)
        editor = self.query_one("#value-input", Input)

        if param.type in (CfgType.DOUBLE, CfgType.INT):
            result = validate_numeric_text(param, editor.value)
            if not result.ok:
                self.query_one("#validation", Static).update(result.error)
                editor.focus()
                return False
            self._commit_value(ref, result.value)
            return True

        if param.type == CfgType.QSTRING:
            if param.max_len > 0 and len(editor.value) > param.max_len:
                self.query_one("#validation", Static).update(
                    f"Expected at most {param.max_len} characters"
                )
                editor.focus()
                return False
            self._commit_value(ref, editor.value)
            return True

        return False

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search":
            self.query_one("#tree", Tree).focus()
            return
        if event.input.id == "value-input" and self._commit_text_editor():
            self.query_one("#tree", Tree).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search":
            self._populate_tree()

    def on_select_changed(self, event: Select.Changed) -> None:
        if (
            self.selected is None
            or event.select.id != "enum-editor"
            or _select_value_is_empty(event.value)
        ):
            return
        if self.current[self.selected.kind][self.selected.name] == event.value:
            return
        self._commit_value(self.selected, int(cast(int, event.value)))

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if self.selected is None or event.switch.id != "bool-editor":
            return
        self._commit_value(self.selected, 1 if event.value else 0)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if self.selected is None or not str(event.checkbox.id or "").startswith("bit-"):
            return
        value = 0
        for index, checkbox in enumerate(self._checkboxes):
            if checkbox.value:
                value |= 1 << index
        self._commit_value(self.selected, value)

    def _step_selected(self, direction: int, multiplier: int = 1) -> bool:
        if self.selected is None:
            return False
        param = self._param(self.selected)
        if param.type not in (CfgType.DOUBLE, CfgType.INT):
            return False
        value = self.current[self.selected.kind][self.selected.name]
        next_value, clamped = step_numeric_value(param, value, direction, multiplier)
        self._commit_value(self.selected, next_value)
        if clamped:
            self._set_status("Value clamped to XML range")
        return True

    def _focus_next_skip_search(self, direction: int) -> bool:
        focus_chain = self.screen.focus_chain
        if not focus_chain:
            return False

        focused = self.focused
        try:
            start = focus_chain.index(focused) if focused is not None else -1
        except ValueError:
            start = -1

        for offset in range(1, len(focus_chain) + 1):
            candidate = focus_chain[(start + direction * offset) % len(focus_chain)]
            if getattr(candidate, "id", None) != "search":
                candidate.focus()
                return True
        return False

    def on_key(self, event: events.Key) -> None:
        focused = self.focused
        if event.key in {"tab", "shift+tab"}:
            direction = -1 if event.key == "shift+tab" else 1
            if self._focus_next_skip_search(direction):
                event.prevent_default()
                event.stop()
            return

        if focused is not None and getattr(focused, "id", None) == "value-input":
            if event.key == "up" and self._step_selected(1):
                event.stop()
            elif event.key == "down" and self._step_selected(-1):
                event.stop()
            elif event.key == "pageup" and self._step_selected(1, 10):
                event.stop()
            elif event.key == "pagedown" and self._step_selected(-1, 10):
                event.stop()
            elif event.key == "escape":
                self._show_selected()
                self.query_one("#tree", Tree).focus()
                event.stop()
        elif focused is not None and getattr(focused, "id", None) == "search":
            if event.key == "escape":
                self.query_one("#search", Input).value = ""
                self._populate_tree()
                self.query_one("#tree", Tree).focus()
                event.stop()

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    def action_toggle_theme(self) -> None:
        self.action_toggle_dark()
        mode = "dark" if self.theme == "textual-dark" else "light"
        self._set_status(f"Theme: {mode}")

    def action_switch_config(self, kind: str) -> None:
        if kind not in {"mcconf", "appconf"}:
            return
        self.active_kind = cast(ConfigKind, kind)
        self._refresh_tabs()
        tree = self.query_one("#tree", Tree)
        tree.focus()
        tree.move_cursor(self._config_nodes.get(self.active_kind), animate=True)
        self._set_status(f"Focused {'motor' if kind == 'mcconf' else 'app'} config")

    def action_revert_selected(self) -> None:
        if self.selected is None:
            self._set_status("No selected field to revert")
            return
        ref = self.selected
        self.current[ref.kind][ref.name] = self.original[ref.kind][ref.name]
        node = self._tree_labels.get(ref)
        if node is not None:
            node.set_label(self._node_label(ref))
        self._show_selected()
        self._set_status("Reverted selected field")

    def action_revert_all(self) -> None:
        if not self._dirty_refs():
            self._set_status("No changes to revert")
            return
        self.push_screen(
            ConfirmModal("Revert all changes?", "Discard every unsaved edit in memory.", "Revert"),
            self._revert_all_decision,
        )

    def _revert_all_decision(self, confirmed: bool) -> None:
        if not confirmed:
            return
        self.current = {
            "mcconf": dict(self.original["mcconf"]),
            "appconf": dict(self.original["appconf"]),
        }
        self._populate_tree()
        self._show_selected()
        self._set_status("Reverted all changes")

    def _change_rows(self) -> list[tuple[str, str, str, str, str]]:
        rows: list[tuple[str, str, str, str, str]] = []
        for ref in self._dirty_refs():
            param = self._param(ref)
            rows.append(
                (
                    "Motor" if ref.kind == "mcconf" else "App",
                    param.long_name or ref.name,
                    ref.name,
                    format_value(param, self.original[ref.kind][ref.name]),
                    format_value(param, self.current[ref.kind][ref.name]),
                )
            )
        return rows

    def action_apply_changes(self) -> None:
        rows = self._change_rows()
        if not rows:
            self._set_status("No changes to apply")
            return
        self.push_screen(ApplyChangesModal(rows), self._apply_decision)

    def _apply_decision(self, confirmed: bool) -> None:
        if not confirmed:
            return
        self._set_status("Writing...")
        try:
            if any(ref.kind == "mcconf" for ref in self._dirty_refs()):
                self.client.set_mcconf(self.current["mcconf"], wait_ack=True)
            if any(ref.kind == "appconf" for ref in self._dirty_refs()):
                self.client.set_appconf(self.current["appconf"], store=True, wait_ack=True)
        except Exception as exc:
            self._set_status(f"Write failed: {exc}")
            return

        self.original = {
            "mcconf": dict(self.current["mcconf"]),
            "appconf": dict(self.current["appconf"]),
        }
        self._populate_tree()
        self._show_selected()
        self._set_status("Applied successfully")

    def action_request_quit(self) -> None:
        if self._dirty_refs():
            self.push_screen(
                ConfirmModal(
                    "Discard unsaved changes?",
                    "Unsaved edits will not be written to the VESC.",
                    "Discard",
                ),
                self._quit_decision,
            )
        else:
            self.exit()

    def _quit_decision(self, confirmed: bool) -> None:
        if confirmed:
            self.exit()

    def action_help(self) -> None:
        self.push_screen(HelpModal())


def _connect_and_load(
    host: str,
    port: int,
    timeout: float,
    config_dir: Path | None,
) -> tuple[VescClient, ConfigSchema, ConfigSchema, dict[str, object], dict[str, object]]:
    client = VescClient.connect_tcp(host, port, timeout=timeout, config_dir=config_dir)
    mc_schema = client.mcconf_schema
    app_schema = client.appconf_schema
    if mc_schema is None:
        client.close()
        raise RuntimeError("No local MCCONF schema found for connected firmware")
    if app_schema is None:
        client.close()
        raise RuntimeError("No local APPCONF schema found for connected firmware")

    mc_values = client.get_mcconf()
    app_values = client.get_appconf()
    return client, mc_schema, app_schema, mc_values, app_values


def _connection_refused_message(endpoint: str, port: int) -> str:
    return "\n".join(
        [
            f"Could not connect to VESC Tool TCP server at {endpoint}: connection refused.",
            "",
            "Start VESC Tool with tcpServer enabled, then retry. For example:",
            f"  vesc_tool --offscreen --vescPort /dev/ttyACM0 --tcpServer {port}",
            "",
            "If VESC Tool is already running, check the host and port passed to --tcp.",
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Edit VESC motor/app configs over tcpServer")
    parser.add_argument("--tcp", type=_parse_tcp_endpoint, required=True, metavar="HOST:PORT")
    parser.add_argument("--timeout", type=float, default=2.0, metavar="SEC")
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", help="Reserved for Textual debugging")
    args = parser.parse_args(argv)

    host, port = args.tcp
    endpoint = f"{host}:{port}"
    client: VescClient | None = None
    try:
        try:
            client, mc_schema, app_schema, mc_values, app_values = _connect_and_load(
                host, port, args.timeout, args.config_dir
            )
        except ConnectionRefusedError:
            raise SystemExit(_connection_refused_message(endpoint, port)) from None
        app = ConfigTuiApp(
            client=client,
            endpoint=endpoint,
            mc_schema=mc_schema,
            app_schema=app_schema,
            mc_values=mc_values,
            app_values=app_values,
        )
        app.run()
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
