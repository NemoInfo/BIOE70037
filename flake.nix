{
  description = "Python for BIOE70037 course flake";

  inputs = { nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable"; };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs.python312Packages; [
          pkgs.jupyter-all
          scipy
          openpyxl
          python
          numpy
          pandas
          matplotlib
        ];

        shellHook = ''
          THEME="af-magic" exec $SHELL 
        '';
      };
    };
}
