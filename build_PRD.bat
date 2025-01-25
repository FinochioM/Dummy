@echo off

sokol-shdc -i dummy/shader.glsl -o dummy/shader.odin -l hlsl5:wgsl -f sokol_odin --save-intermediate-spirv

if not exist build_release mkdir build_release

pushd build_release
odin build ../dummy -o:speed
popd