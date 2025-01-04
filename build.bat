@echo off

rem https://github.com/floooh/sokol-tools/blob/master/docs/sokol-shdc.md
sokol-shdc -i dummy/shader.glsl -o dummy/shader.odin -l hlsl5:wgsl -f sokol_odin

odin build dummy -debug