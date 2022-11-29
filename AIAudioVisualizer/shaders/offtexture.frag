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

float getLevel(float x, int type) {
	if (type == 0)
	{
		return sceneData.audiodata01.x;
	}
	if (type == 1)
	{
		return sceneData.audiodata01.y;
	}
	if (type == 2)
	{
		return sceneData.audiodata01.z;
	}
	if (type == 3)
	{
		return sceneData.audiodata01.w;
	}
}

float getPitch(float freq, int octave, int type){
   return getLevel(pow(2.0, float(octave)) * freq / 12000.0, type);
}

float logX(float x, float a, float c){
   return 1.0 / (exp(-a*(x-c)) + 1.0);
}

float logisticAmp(float amp){
   float c = 1.0 - (0.25);
   float a = 20.0 * (1.0 - 0.5);  
   a = 20.;   
   return (logX(amp, a, c) - logX(0.0, a, c)) / (logX(1.0, a, c) - logX(0.0, a, c));
}

float getAudioIntensityAt(float x, int type) {
    x = abs(fract(x));
    float freq = pow(2., x*3.) * 261.;
    return logisticAmp(getPitch(freq, 1, type));
}

void main() 
{	
	//vec2 coord = vec2(fragCoord.x, fragCoord.y + sin(sceneData.frame * 5.0f));
	//vec2 coord = vec2(fragCoord.x, fragCoord.y + sceneData.audiodata01.x * 1000.0f);
	
	float wavepoint = smoothstep( 0.0, 0.00001, sceneData.audiodata01.x) - 1.0f;
	//float wavepoint = step(0.0f, abs(sceneData.audiodata01.x)) - 1.0f;
	//float wavepoint = getAudioIntensityAt(0.3, 0);

	vec2 coord = vec2(fragCoord.x, fragCoord.y + wavepoint);
	float dis = distance(coord, vec2(-0.1f, 6.0f));
	vec3 finalcolor = vec3(0.0f, 0.0f, 0.0f);

	vec2 offtex01 = vec2(texCoord.x + 0.005f, -texCoord.y);
	vec2 offtex02 = vec2(texCoord.x - 0.005f, -texCoord.y);
	vec3 oldcolor01 = texture(tex1, offtex01).xyz;
	vec3 oldcolor02 = texture(tex1, offtex02).xyz;

	// add wave form on top	
	//float wavecol = 1.0 -  smoothstep( 0.0, 0.15, abs(sceneData.audiodata01.x - texCoord.y) );

	if (dis < 0.2){
		finalcolor = vec3(1.0f, 0.0f,0.0f);
	}

	wavepoint = smoothstep( 0.0, 0.00001, sceneData.audiodata01.y) - 1.0f;
	//wavepoint = getAudioIntensityAt(0.4, 1);

	coord = vec2(fragCoord.x, fragCoord.y + wavepoint);
	dis = distance(coord, vec2(-0.1f, 3.0f));
	
	if (dis < 0.2){
		finalcolor = vec3(0.0f, 1.0f,0.0f);
	}

	wavepoint = smoothstep( 0.0, 0.00001, sceneData.audiodata01.z) - 1.0f;
	//wavepoint = getAudioIntensityAt(0.5, 2);

	coord = vec2(fragCoord.x, fragCoord.y + wavepoint);
	dis = distance(coord, vec2(-0.1f, 0.0f));
	
	if (dis < 0.2){
		finalcolor = vec3(0.0f, 0.0f,1.0f);
	}

	wavepoint = smoothstep( 0.0, 0.00001, sceneData.audiodata01.w) - 1.0f;
	//wavepoint = getAudioIntensityAt(0.7, 3);

	coord = vec2(fragCoord.x, fragCoord.y + wavepoint);
	dis = distance(coord, vec2(-0.1f, -3.0f));
	
	if (dis < 0.2){
		finalcolor = vec3(0.0f, 1.0f,1.0f);
	}

	//finalcolor.x = wavecol;
	finalcolor = finalcolor  + oldcolor01 * 0.45 + oldcolor02 * 0.45 ;

	outFragColor = vec4(finalcolor, 1.0f);
}