//glsl version 4.5
#version 460

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
	float frame;
	float shaderoffset;
} sceneData;

layout(set = 2, binding = 0) uniform sampler2D tex1;

struct FFTData{
	vec4 audio;
};

//all object matrices
layout(std140,set = 1, binding = 1) readonly buffer ObjectBuffer{

	FFTData audio[];
} fftBuffer;

void main() 
{	
	//float wavepoint = smoothstep( 0.0, 0.00001, sceneData.audiodata01.x) - 1.0f;

	vec2 coord = vec2(fragCoord.x, fragCoord.y);
	float dis = distance(0.0f, coord.x);
	vec3 finalcolor = vec3(0.0f, 0.0f, 0.0f);

	vec2 offtex01 = vec2(texCoord.x + 0.01f, -texCoord.y);
	vec2 offtex02 = vec2(texCoord.x - 0.01f, -texCoord.y);
	vec3 oldcolor01 = texture(tex1, offtex01).xyz;
	vec3 oldcolor02 = texture(tex1, offtex02).xyz;

	if (dis < 0.2){
		float ydist = distance(0.0, coord.y - 6.0);

		if (ydist < 1.5)
		{
			uint ind1 = uint(clamp(abs(fragCoord.y - 6.0),0.0,1.5) / 1.5 * 32.0);
			float valred = fftBuffer.audio[ind1].audio.x;
			finalcolor.r = valred;
		}

		ydist = distance(0.0, coord.y - 3.0);

		if (ydist < 1.5)
		{
			uint ind2 = uint(clamp(abs(fragCoord.y - 3.0),0.0,1.5) / 1.5 * 32.0);
			float valgreen= fftBuffer.audio[ind2].audio.y;
			finalcolor.g = valgreen;
		}

		ydist = distance(0.0, coord.y );

		if (ydist < 1.5)
		{
			uint ind3 = uint(clamp(abs(fragCoord.y),0.0,1.5) / 1.5 * 32.0) ;
			float valblue= fftBuffer.audio[ind3].audio.z;
			finalcolor.b = valblue;
		}

		ydist = distance(0.0, coord.y + 3.0);

		if (ydist < 1.5)
		{
			uint ind4 = uint(clamp(abs(fragCoord.y + 3.0),0.0,1.5) / 1.5 * 32.0) ;
			float valyell= fftBuffer.audio[ind4].audio.w;
			finalcolor.r = valyell;
			finalcolor.g = valyell;
		}
	}

	//finalcolor.x = wavecol;
	finalcolor = finalcolor  + oldcolor01 * 0.48 * sceneData.shaderoffset + oldcolor02 * 0.48 * sceneData.shaderoffset  ;

	outFragColor = vec4(finalcolor, 1.0f);
}