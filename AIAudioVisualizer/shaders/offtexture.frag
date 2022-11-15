//glsl version 4.5
#version 450

//shader input
layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 texCoord;
layout (location = 2) in vec3 fragCoord;

//output write
layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 1) uniform  SceneData{   
    vec4 fogColor; // w is for exponent
	vec4 fogDistances; //x for min, y for max, zw unused.
	vec4 ambientColor;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
} sceneData;

layout(set = 2, binding = 0) uniform sampler2D tex1;

void main() 
{	
	//outFragColor = vec4(inColor + sceneData.ambientColor.xyz,1.0f);
	//vec3 color = texture(tex1,texCoord).xyz;

	float dis = distance(fragCoord.xy, vec2(0.5f));

	if (dis < 3.2){
		//outFragColor = vec4(0.5f,0.5f,0.5f,1.0f);
		vec2 offtex = vec2(texCoord.x - 0.0001f, texCoord.y);
		vec3 color = texture(tex1, texCoord).xyz;
		outFragColor = vec4(color.x, color.y, color.z, 1.0f);
	}
	else
	{
		outFragColor = vec4(1.0f,1.0f,0.0f,1.0f);
	}
}