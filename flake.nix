{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python313;
          # PyPI wheels (numpy, PySide6, ...) link system libraries; NixOS has no default FHS path.
          linuxPyPiLdPath = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.fontconfig
              pkgs.freetype
              pkgs.dbus
              pkgs.glib
              pkgs.libglvnd
              pkgs.libxkbcommon
              pkgs.libxcb
              pkgs.libxcb-cursor
              pkgs.libxcb-image
              pkgs.libxcb-keysyms
              pkgs.libxcb-render-util
              pkgs.libxcb-util
              pkgs.xcbutilwm
              pkgs.stdenv.cc.cc.lib
              pkgs.libice
              pkgs.libsm
              pkgs.libx11
              pkgs.libxext
              pkgs.libxi
              pkgs.libxrender
              pkgs.wayland
              pkgs.zlib
              pkgs.zstd
            ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
          pyShell = pkgs.mkShell {
            packages = [ pkgs.uv python ];
            shellHook = linuxPyPiLdPath + ''
              export UV_PYTHON="${python}/bin/python3"
              # Run `nix develop` from this directory (same folder as pyproject.toml), not via `cd` into the flake store path.
              if [[ -f pyproject.toml ]] && [[ ! -d .venv ]]; then
                echo "vesc-py dev shell: creating .venv."
                rm -rf .venv
                uv venv
                uv sync
              fi
            '';
          };
        in {
          default = pyShell;
          python = pyShell;
        }
      );
    };
}
