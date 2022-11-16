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
	vec4 audiodata01;
	vec4 audiodata02;
	float frame;
} sceneData;

layout(set = 2, binding = 0) uniform sampler2D tex1;

void main() 
{	
	//vec2 coord = vec2(fragCoord.x, fragCoord.y + sin(sceneData.frame * 5.0f));
	vec2 coord = vec2(fragCoord.x, fragCoord.y + sceneData.audiodata01.x * 1000.0f);
	float dis = distance(coord, vec2(0.5f));
	vec3 finalcolor = vec3(0.0f, 0.0f, 0.0f);
	vec2 offtex = vec2(texCoord.x + 0.005f, -texCoord.y);
	vec3 oldcolor = texture(tex1, offtex).xyz;


	if (dis < 0.2){
		finalcolor = vec3(1.0f, 0.0f,0.0f);
	}
	
	finalcolor = finalcolor  + oldcolor * 0.9;

	outFragColor = vec4(finalcolor, 1.0f);
}