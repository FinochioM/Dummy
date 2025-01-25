#version 450
#define SOKOL_HLSL (1)
in vec2 position;
in vec4 color0;
in vec2 uv0;
in vec4 bytes0;
in vec4 color_override0;

out vec4 color;
out vec2 uv;
out vec4 bytes;
out vec4 color_override;


void main() {
	gl_Position = vec4(position, 0, 1);
	color = color0;
	uv = uv0;
	bytes = bytes0;
	color_override = color_override0;
}
