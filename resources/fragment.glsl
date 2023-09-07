
#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;

// texture sampler
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float mixValue;

void main()
{
	//vec4 t0 = texture(texture0, TexCoord);
	//vec4 t1 = texture(texture1, TexCoord);
	FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	//FragColor = vec4(ourColor, 1.0f);
}


