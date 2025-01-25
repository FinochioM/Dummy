#version 450
#define SOKOL_HLSL (1)
layout(binding=0) uniform texture2D tex0;
layout(binding=1) uniform texture2D tex1;
layout(binding=0) uniform sampler default_sampler;

in vec4 color;
in vec2 uv;
in vec4 bytes;
in vec4 color_override;

out vec4 col_out;

void main() {

	int tex_index = int(bytes.x * 255.0);
	
	vec4 tex_col = vec4(1.0);
	if (tex_index == 0) {
		tex_col = texture(sampler2D(tex0, default_sampler), uv);
	} else if (tex_index == 1) {
		                                                                                  
		tex_col.a = texture(sampler2D(tex1, default_sampler), uv).r;
	}
	
	col_out = tex_col;
	col_out *= color;
	
	col_out.rgb = mix(col_out.rgb, color_override.rgb, color_override.a);
}
