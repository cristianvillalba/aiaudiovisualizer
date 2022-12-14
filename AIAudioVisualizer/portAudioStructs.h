#ifndef PORTAUDIOSTRUCT_H
#define PORTAUDIOSTRUCT_H


#include <vector>
#include <string>
#include <deque>

#define SAMPLE_RATE  (44100)
#define FRAMES_PER_BUFFER (512)
#define NUM_SECONDS     (0.3)
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
	void* waveout;

	std::vector<std::string> * devices;
	int deviceselection;
	float shaderFrameOffset;
};

class VisualData
{
public:
	std::deque<float> visualbuffer00;
	std::deque<float> visualbuffer01;
	std::deque<float> visualbuffer02;
	std::deque<float> visualbuffer03;

	std::deque<float> visualbufferfft00;
	std::deque<float> visualbufferfft01;
	std::deque<float> visualbufferfft02;
	std::deque<float> visualbufferfft03;
};

class VisualDataWrapper
{
private:
	VisualData visualData;
	std::mutex m;
public:
	int getBuffer00Size()
	{
		std::lock_guard<std::mutex> lock(m);
		int size = visualData.visualbuffer00.size();

		return size;
	}

	int getBuffer01Size()
	{
		std::lock_guard<std::mutex> lock(m);
		int size = visualData.visualbuffer01.size();

		return size;
	}

	int getBuffer02Size()
	{
		std::lock_guard<std::mutex> lock(m);
		int size = visualData.visualbuffer02.size();

		return size;
	}

	int getBuffer03Size()
	{
		std::lock_guard<std::mutex> lock(m);
		int size = visualData.visualbuffer03.size();

		return size;
	}

	int getBufferFFT00Size()
	{
		std::lock_guard<std::mutex> lock(m);
		int size = visualData.visualbufferfft00.size();

		return size;
	}

	int getBufferFFT01Size()
	{
		std::lock_guard<std::mutex> lock(m);
		int size = visualData.visualbufferfft01.size();

		return size;
	}

	int getBufferFFT02Size()
	{
		std::lock_guard<std::mutex> lock(m);
		int size = visualData.visualbufferfft02.size();

		return size;
	}

	int getBufferFFT03Size()
	{
		std::lock_guard<std::mutex> lock(m);
		int size = visualData.visualbufferfft03.size();

		return size;
	}

	float getBuffer00Val()
	{
		std::lock_guard<std::mutex> lock(m);
		float val = visualData.visualbuffer00.front();
		visualData.visualbuffer00.pop_front();
		//printf("reading01:%.8f\n", val);
		return val;
	}

	float getBuffer01Val()
	{
		std::lock_guard<std::mutex> lock(m);
		float val = visualData.visualbuffer01.front();
		visualData.visualbuffer01.pop_front();
		//printf("reading02:%.8f\n", val);
		return val;
	}

	float getBuffer02Val()
	{
		std::lock_guard<std::mutex> lock(m);
		float val = visualData.visualbuffer02.front();
		visualData.visualbuffer02.pop_front();
		//printf("reading03:%.8f\n", val);
		return val;
	}

	float getBuffer03Val()
	{
		std::lock_guard<std::mutex> lock(m);
		float val = visualData.visualbuffer03.front();
		visualData.visualbuffer03.pop_front();
		//printf("reading04:%.8f\n", val);
		return val;
	}

	float getBufferFFT00Val()
	{
		std::lock_guard<std::mutex> lock(m);
		if (visualData.visualbufferfft00.size() > 0){
			float val = visualData.visualbufferfft00.front();
			visualData.visualbufferfft00.pop_front();
			//printf("reading01:%.8f\n", val);
			return val;
		}
		else
		{
			return 0.0f;
		}
	}

	float getBufferFFT01Val()
	{
		std::lock_guard<std::mutex> lock(m);
		if (visualData.visualbufferfft01.size() > 0) {
			float val = visualData.visualbufferfft01.front();
			visualData.visualbufferfft01.pop_front();
			//printf("reading02:%.8f\n", val);
			return val;
		}
		else
		{
			return 0.0f;
		}
	}

	float getBufferFFT02Val()
	{
		std::lock_guard<std::mutex> lock(m);
		if (visualData.visualbufferfft02.size() > 0) {
			float val = visualData.visualbufferfft02.front();
			visualData.visualbufferfft02.pop_front();
			//printf("reading03:%.8f\n", val);
			return val;
		}
		else
		{
			return 0.0f;
		}
	}

	float getBufferFFT03Val()
	{
		std::lock_guard<std::mutex> lock(m);
		if (visualData.visualbufferfft03.size() > 0) {
			float val = visualData.visualbufferfft03.front();
			visualData.visualbufferfft03.pop_front();
			//printf("reading04:%.8f\n", val);
			return val;
		}
		else
		{
			return 0.0f;
		}
	}

	void setBuffer00Val(float val, int maxbuff)
	{
		std::lock_guard<std::mutex> lock(m);
		if (!isnan(val) && abs(val) <= 1.0f && visualData.visualbuffer00.size() < maxbuff) {
			//printf("writing:%.8f\n", val);
			visualData.visualbuffer00.push_back(val);
		}
	}

	void setBuffer01Val(float val, int maxbuff)
	{
		std::lock_guard<std::mutex> lock(m);
		if (!isnan(val) && abs(val) <= 1.0f && visualData.visualbuffer01.size() < maxbuff) {
			//printf("writing:%.8f\n", val);
			visualData.visualbuffer01.push_back(val);
		}
	}

	void setBuffer02Val(float val, int maxbuff)
	{
		std::lock_guard<std::mutex> lock(m);
		if (!isnan(val) && abs(val) <= 1.0f && visualData.visualbuffer02.size() < maxbuff) {
			//printf("writing:%.8f\n", val);
			visualData.visualbuffer02.push_back(val);
		}
	}

	void setBuffer03Val(float val, int maxbuff)
	{
		std::lock_guard<std::mutex> lock(m);
		if (!isnan(val) && abs(val) <= 1.0f && visualData.visualbuffer03.size() < maxbuff) {
			//printf("writing:%.8f\n", val);
			visualData.visualbuffer03.push_back(val);
		}
	}

	void setBufferFFT00Val(float val, int maxbuff)
	{
		std::lock_guard<std::mutex> lock(m);
		if (!isnan(val) && visualData.visualbufferfft00.size() < maxbuff) {
			//printf("writing:%.8f\n", val);
			visualData.visualbufferfft00.push_back(val);
		}
	}

	void setBufferFFT01Val(float val, int maxbuff)
	{
		std::lock_guard<std::mutex> lock(m);
		if (!isnan(val) && visualData.visualbufferfft01.size() < maxbuff) {
			//printf("writing:%.8f\n", val);
			visualData.visualbufferfft01.push_back(val);
		}
	}

	void setBufferFFT02Val(float val, int maxbuff)
	{
		std::lock_guard<std::mutex> lock(m);
		if (!isnan(val) && visualData.visualbufferfft02.size() < maxbuff) {
			//printf("writing:%.8f\n", val);
			visualData.visualbufferfft02.push_back(val);
		}
	}

	void setBufferFFT03Val(float val, int maxbuff)
	{
		std::lock_guard<std::mutex> lock(m);
		if (!isnan(val) && visualData.visualbufferfft03.size() < maxbuff) {
			//printf("writing:%.8f\n", val);
			visualData.visualbufferfft03.push_back(val);
		}
	}

	void setBufferFFT00Val(float * buff, int maxbuff, int fftmaxbuf)
	{
		std::lock_guard<std::mutex> lock(m);

		for (int i = 0; i < maxbuff; i++)
		{
			visualData.visualbufferfft00.push_back(buff[i]);

			if (visualData.visualbufferfft00.size() >= fftmaxbuf)
			{
				visualData.visualbufferfft00.pop_front();
			}
		}
	}

	void setBufferFFT01Val(float* buff, int maxbuff, int fftmaxbuf)
	{
		std::lock_guard<std::mutex> lock(m);

		for (int i = 0; i < maxbuff; i++)
		{
			visualData.visualbufferfft01.push_back(buff[i]);

			if (visualData.visualbufferfft01.size() >= fftmaxbuf)
			{
				visualData.visualbufferfft01.pop_front();
			}
		}
	}

	void setBufferFFT02Val(float* buff, int maxbuff, int fftmaxbuf)
	{
		std::lock_guard<std::mutex> lock(m);

		for (int i = 0; i < maxbuff; i++)
		{
			visualData.visualbufferfft02.push_back(buff[i]);

			if (visualData.visualbufferfft02.size() >= fftmaxbuf)
			{
				visualData.visualbufferfft02.pop_front();
			}
		}
	}

	void setBufferFFT03Val(float* buff, int maxbuff, int fftmaxbuf)
	{
		std::lock_guard<std::mutex> lock(m);

		for (int i = 0; i < maxbuff; i++)
		{
			visualData.visualbufferfft03.push_back(buff[i]);

			if (visualData.visualbufferfft03.size() >= fftmaxbuf)
			{
				visualData.visualbufferfft03.pop_front();
			}
		}
	}

	void getBuffer00FFTVals(glm::float32 * buff)
	{
		std::lock_guard<std::mutex> lock(m);
	
		for (int i = 0; visualData.visualbufferfft00.size() > 0; i++)
		{
			float val = visualData.visualbufferfft00.front();
			visualData.visualbufferfft00.pop_front();
			buff[i] = val;
		}
	}

	void getBuffer01FFTVals(glm::float32* buff)
	{
		std::lock_guard<std::mutex> lock(m);

		for (int i = 0; visualData.visualbufferfft01.size() > 0; i++)
		{
			float val = visualData.visualbufferfft01.front();
			visualData.visualbufferfft01.pop_front();
			buff[i] = val;
		}
	}

	void getBuffer02FFTVals(glm::float32* buff)
	{
		std::lock_guard<std::mutex> lock(m);

		for (int i = 0; visualData.visualbufferfft02.size() > 0; i++)
		{
			float val = visualData.visualbufferfft02.front();
			visualData.visualbufferfft02.pop_front();
			buff[i] = val;
		}
	}

	void getBuffer03FFTVals(glm::float32* buff)
	{
		std::lock_guard<std::mutex> lock(m);

		for (int i = 0; visualData.visualbufferfft03.size() > 0; i++)
		{
			float val = visualData.visualbufferfft03.front();
			visualData.visualbufferfft03.pop_front();
			buff[i] = val;
		}
	}

	
};

extern struct paTestData audioData; //this instance is defined in vk_engine.cpp
extern class VisualDataWrapper visualDataWrapper; //this instance is defined in vk_engine.cpp


#endif