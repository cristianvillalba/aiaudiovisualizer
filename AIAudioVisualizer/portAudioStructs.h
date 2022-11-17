#ifndef PORTAUDIOSTRUCT_H
#define PORTAUDIOSTRUCT_H


#include <vector>
#include <string>
#include <deque>

#define SAMPLE_RATE  (44100)
#define FRAMES_PER_BUFFER (512)
#define NUM_SECONDS     (1)
#define NUM_CHANNELS    (2)
/* #define DITHER_FLAG     (paDitherOff) */
#define DITHER_FLAG     (0) /**/
#define WRITE_TO_FILE   (0)

/* Sample format. */
#define PA_SAMPLE_TYPE  paFloat32
typedef float SAMPLE;
#define SAMPLE_SILENCE  (0.0f)
#define PRINTF_S_FORMAT "%.8f"

struct paTestData
{
	int          frameIndex;  /* Index into sample array. */
	int          maxFrameIndex;
	int			 visualoffset;
	SAMPLE* recordedSamples;
	void* aipredicter;
	float* bufferpredictl00;
	float* bufferpredictr00;
	float* bufferpredictl01;
	float* bufferpredictr01;
	float* bufferpredictl02;
	float* bufferpredictr02;
	float* bufferpredictl03;
	float* bufferpredictr03;
	void* waveout;

	std::vector<std::string> * devices;
	int deviceselection;

	std::deque<float>* visualbuffer00;
	std::deque<float>* visualbuffer01;
	std::deque<float>* visualbuffer02;
	std::deque<float>* visualbuffer03;
};

extern struct paTestData audioData; //this instance is defined in vk_engine.cpp

#endif