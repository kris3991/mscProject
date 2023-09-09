
#version 330 core
out vec4 FragColor;

in vec3 outputColor;
in vec2 TexCoord;

// texture sampler
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float mixValue;

void main()
{
	FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}


