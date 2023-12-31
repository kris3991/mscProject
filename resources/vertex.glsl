#version 330 core

//inputs.
in vec3 aPosition;
in vec2 aTexCoord;
in vec3 aNormal;

//outputs.
out vec3 outputColor;
out vec2 TexCoord;

//uniforms.
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

//textures are not really used, but just maintaining the vao structure.

void main()
{
	mat4 transformation = mat4(projection * view * model);
	gl_Position = transformation * vec4(aPosition, 1.0);
	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
	outputColor = vec3(1.0f, 1.0f, 1.0f);
}
