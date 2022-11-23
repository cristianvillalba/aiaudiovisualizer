#include <AIAudioVisualizer.h>
#include <vk_engine.h>

VulkanEngine engine;
std::mutex BigMutex; //defining big mutex class

int main(int argc, char* argv[])
{
	engine.init();

	engine.run();

	engine.cleanup();

	return 0;
}

void AudioVisualizer::init()
{
	std::cout << "Init TensorFlow C library version " << TF_Version() << std::endl;
	logAI.AddLog("[%s] - %s %s\n", "info", "Init TensorFlow C library version", TF_Version());
}

int AudioVisualizer::loadModel()
{
	model = new cppflow::model(std::string("model"));
	auto operations = model->get_operations();

	std::cout << "Load graph success" << std::endl;
	logAI.AddLog("[%s] - %s\n","info", "Load graph success");

	std::cout << "Checking operations..." << std::endl;
	logAI.AddLog("[%s] - %s\n","info", "Checking operations...");

	for (int i = 0; i < operations.size(); i++)
	{
		std::cout << operations[i] << std::endl;

		logAI.AddLog("[%s] - %s\n", "info", operations[i].c_str());
	}
	return 0;
}

int AudioVisualizer::initSound(int samplesperchannel)
{
	//AudioFile<float> audioFile;
	//audioFile.load("short.wav");

	//audioFile.printSummary();
	//std::cout << "Load sound success" << std::endl;

	scale = 2.0000630488853100000;

	size_t size = 1 * 1025 * 21 * 2; //final size of tensor input
	std::vector<float> _data(size, 0.0f);    // make room for tensor and initialize them to 0
	std::vector<int64_t> shape{ 1 , 1025, 21, 2 };


	stftl = new gam::STFT{
		2048,		// Window size
		512 ,		// Hop size; number of samples between transforms
		0,			// Pad size; number of zero-valued samples appended to window
		gam::WindowType::HANN,		// Window type: BARTLETT, BLACKMAN, BLACKMAN_HARRIS,
					//		HAMMING, HANN, WELCH, NYQUIST, or RECTANGLE
		gam::SpectralType::COMPLEX		// Format of frequency samples:
					//		COMPLEX, MAG_PHASE, or MAG_FREQ
	};

	stftr = new gam::STFT{
		2048,		// Window size
		512 ,		// Hop size; number of samples between transforms
		0,			// Pad size; number of zero-valued samples appended to window
		gam::WindowType::HANN,		// Window type: BARTLETT, BLACKMAN, BLACKMAN_HARRIS,
					//		HAMMING, HANN, WELCH, NYQUIST, or RECTANGLE
		gam::SpectralType::COMPLEX		// Format of frequency samples:
					//		COMPLEX, MAG_PHASE, or MAG_FREQ
	};

	numSamples = samplesperchannel;
	nframes = (int)numSamples / 512 + 2; //hop size + 2 because it's padded with 0s
	
	superarray = xt::xarray<float>::from_shape({ 1 , 1025, (unsigned long) nframes, 2 }); //all frames
	arraytransposed = xt::xarray<float>::from_shape({ 1025, (unsigned long)nframes, 2 });
	phasetransposed = xt::xarray<float>::from_shape({ 1025, (unsigned long)nframes, 2 });

	stftpredicted = xt::zeros<double>({ 1025, nframes, 2, 4 });
	softmasks = xt::xarray<double>::from_shape({ 1025, (unsigned long)nframes, 2, 4 });
	sources = xt::xarray<double>::from_shape({ 1025, (unsigned long)nframes, 2 , 4 });

	for (int i = 0; i < 1536; i++)
	{
		stftl->operator()(0.0);
		stftr->operator()(0.0);
	}

	
	
	return 0;
}

int AudioVisualizer::predict(float* samples, int numchannels, float* bufflstart00, float* buffrstart00, float* bufflstart01, float* buffrstart01, float* bufflstart02, float* buffrstart02, float* bufflstart03, float* buffrstart03, std::deque<float>* visualbuffer00, std::deque<float>* visualbuffer01, std::deque<float>* visualbuffer02, std::deque<float>* visualbuffer03)
{
	currentFrame = 0;
	size_t size = 1 * 1025 * 21 * 2; //final size of tensor input
	std::vector<float> _data(size, 0.0f);    // make room for tensor and initialize them to 0

	/*for (int i = 0; i < 512; i++)
	{
		stftl->operator()(samples[i * numchannels]);
		stftr->operator()(samples[(i * numchannels) + 1]);
	}*/

	//for (int i = 512; i < numSamples; i++)  
	for (int i = 0; i < numSamples; i++)
	{
		float currentSample = samples[i * numchannels];

		if (stftl->operator()(currentSample)) {
			// Loop through all the bins
			for (unsigned k = 0; k < stftl->numBins(); ++k) {
				superarray(0, k, currentFrame, 0) = log2(stftl->bin(k).mag() * scale + 1.0);
				arraytransposed(k, currentFrame, 0) = stftl->bin(k).mag();
				phasetransposed(k, currentFrame, 0) = stftl->bin(k).phase();
				//{1 * 1025 * 21 * 2}; //shape of tensor
				//int index = a * 1025 * 21 * 2
				//	+ b * 21 * 2
				//	+ c * 2
				//	+ d;
				//int index = 0 * 1025 * 21 * 2 +
				//			k * 21 * 2 +
				//			0 * 2 +
				//			0;
				//_data[index] = log2(stftl.bin(k).mag() + 1.0);

				//std::cout << stftl.bin(k).mag() << " "<< stftl.bin(k).arg() << std::endl;
				//std::cout << stftl.bin(k).real() << "," << stftl.bin(k).imag() << std::endl;
				//std::cout << log2(stftl.bin(k).mag() * scale + 1.0) << std::endl;

			}
		}

		currentSample = samples[(i * numchannels) + 1];

		if (stftr->operator()(currentSample)) {

			// Loop through all the bins
			for (unsigned k = 0; k < stftr->numBins(); ++k) {
				superarray(0, k, currentFrame, 1) = log2(stftr->bin(k).mag() * scale + 1.0);
				arraytransposed(k, currentFrame, 1) = stftr->bin(k).mag();
				phasetransposed(k, currentFrame, 1) = stftr->bin(k).phase();
				//{1 * 1025 * 21 * 2}; //shape of tensor
				//int index = a * 1025 * 21 * 2
				//	+ b * 21 * 2
				//	+ c * 2
				//	+ d;
				//int index = 0 * 1025 * 21 * 2 +
				//	k * 21 * 2 +
				//	0 * 2 +
				//	1;
				//_data[index] = log2(stftr.bin(k).mag() + 1.0);
			}
			currentFrame++;
		}
	}


	double division = 1.0 / 7.0;//dumb rounding bug, integer division was before (1/7 => 0)

	for (auto&& i : xt::arange(10, nframes - 11, 3)) {
		auto frameview = xt::view(superarray, xt::all(), xt::all(), xt::range(i - 10, i + 11), xt::all());

		std::copy(frameview.cbegin(), frameview.cend(), _data.begin());
		cppflow::tensor input = cppflow::tensor(_data, { 1, 1025, 21, 2 });

		auto predicted = model->operator()({ {"serving_default_input_1", input} }, { "StatefulPartitionedCall" }); //solo funciona asi

		std::vector<float> values = predicted[0].get_data<float>();

		std::vector<std::size_t> shapeout = { 1025, 21, 2, 4 };
		auto xtensorpredicted = xt::adapt(values, shapeout);

		auto stftpredictedslice = xt::view(stftpredicted, xt::all(), xt::range(i - 10, i + 11), xt::all(), xt::all());

		stftpredictedslice = stftpredictedslice + division * xtensorpredicted;//dumb rounding bug
	}

	stftpredicted = xt::pow(2, stftpredicted) - 1;
	auto denmask = xt::sum(stftpredicted, { 3 });


	float eps = std::numeric_limits<float>::epsilon();

	for (int j = 0; j < 4; j++)
	{
		auto softmaskslice = xt::view(softmasks, xt::all(), xt::all(), xt::all(), j);
		auto sourcesslice = xt::view(sources, xt::all(), xt::all(), xt::all(), j);
		auto stftpredictedslice = xt::view(stftpredicted, xt::all(), xt::all(), xt::all(), j);

		softmaskslice = stftpredictedslice / (denmask + eps);
		sourcesslice = softmaskslice * arraytransposed;
	}

	xt::xarray<double> magnitudestftoutl = xt::view(sources, xt::all(), xt::all(), 0, xt::all());
	xt::xarray<double> magnitudestftoutr = xt::view(sources, xt::all(), xt::all(), 1, xt::all());

	//-----mix step
	float* bufferindexl00 = bufflstart00;
	float* bufferindexr00 = buffrstart00;
	float* bufferindexl01 = bufflstart01;
	float* bufferindexr01 = buffrstart01;
	float* bufferindexl02 = bufflstart02;
	float* bufferindexr02 = buffrstart02;
	float* bufferindexl03 = bufflstart03;
	float* bufferindexr03 = buffrstart03;

	stftl->inverseWindowing(false);
	stftr->inverseWindowing(false);

	for (int k = 0; k < nframes; k++)
	{
		for (int j = 0; j < stftl->numBins(); j++)
		{
			gam::Complex comp0;
			gam::Complex comp1;

			comp0.fromPolar(magnitudestftoutl(j, k, 0), phasetransposed(j, k, 0));
			comp1.fromPolar(magnitudestftoutr(j, k, 0), phasetransposed(j, k, 1));

			stftl->bin(j).set(comp0);
			stftr->bin(j).set(comp1);
		}

		stftl->inverse(bufferindexl00); //save inverse into buffer
		stftr->inverse(bufferindexr00); //save inverse into buffer

		bufferindexl00 += 512; //hop size
		bufferindexr00 += 512; //hop size

		//----------------------
		for (int j = 0; j < stftl->numBins(); j++)
		{
			gam::Complex comp0;
			gam::Complex comp1;

			comp0.fromPolar(magnitudestftoutl(j, k, 1), phasetransposed(j, k, 0));
			comp1.fromPolar(magnitudestftoutr(j, k, 1), phasetransposed(j, k, 1));

			stftl->bin(j).set(comp0);
			stftr->bin(j).set(comp1);
		}

		stftl->inverse(bufferindexl01); //save inverse into buffer
		stftr->inverse(bufferindexr01); //save inverse into buffer

		bufferindexl01 += 512; //hop size
		bufferindexr01 += 512; //hop size

		//----------------------
		for (int j = 0; j < stftl->numBins(); j++)
		{
			gam::Complex comp0;
			gam::Complex comp1;

			comp0.fromPolar(magnitudestftoutl(j, k, 2), phasetransposed(j, k, 0));
			comp1.fromPolar(magnitudestftoutr(j, k, 2), phasetransposed(j, k, 1));

			stftl->bin(j).set(comp0);
			stftr->bin(j).set(comp1);
		}

		stftl->inverse(bufferindexl02); //save inverse into buffer
		stftr->inverse(bufferindexr02); //save inverse into buffer

		bufferindexl02 += 512; //hop size
		bufferindexr02 += 512; //hop size

		//----------------------
		for (int j = 0; j < stftl->numBins(); j++)
		{
			gam::Complex comp0;
			gam::Complex comp1;

			comp0.fromPolar(magnitudestftoutl(j, k, 3), phasetransposed(j, k, 0));
			comp1.fromPolar(magnitudestftoutr(j, k, 3), phasetransposed(j, k, 1));

			stftl->bin(j).set(comp0);
			stftr->bin(j).set(comp1);
		}

		stftl->inverse(bufferindexl03); //save inverse into buffer
		stftr->inverse(bufferindexr03); //save inverse into buffer

		bufferindexl03 += 512; //hop size
		bufferindexr03 += 512; //hop size
	}

	//std::cout << "prediction done..." << std::endl;

	int buffersize = nframes * 512 + 2048; //adding the last window   

	
	//for (int j = 0; j < (buffersize - 3910); j++) //worthless
	for (int j = (buffersize - 3910); j < buffersize; j++)
	{
		std::lock_guard<std::mutex> lockGuard(BigMutex);
		if (visualbuffer00->size() < 500)
		{
			if (!isnan(bufferindexl00[j]) && abs(bufferindexl00[j]) <= 1.0f) {
				//printf("writing:%.8f\n", bufferindexl00[j]);
				visualbuffer00->push_back(bufferindexl00[j]);
			}
			else
			{
				//visualbuffer00->push_back(0.0);
			}
		}

		if (visualbuffer01->size() < 500)
		{
			if (!isnan(bufferindexl01[j]) && abs(bufferindexl01[j]) <= 1.0f) {
				//printf("writing:%.8f\n", bufferindexl01[j]);
				visualbuffer01->push_back(bufferindexl01[j]);
			}
			else
			{
				//visualbuffer01->push_back(0.0);
			}
		}

		if (visualbuffer02->size() < 500)
		{
			if (!isnan(bufferindexl02[j]) && abs(bufferindexl02[j]) <= 1.0f) {
				//printf("writing:%.8f\n", bufferindexl02[j]);
				visualbuffer02->push_back(bufferindexl02[j]);
			}
			else
			{
				//visualbuffer02->push_back(0.0);
			}
		}

		if (visualbuffer03->size() < 500)
		{
			if (!isnan(bufferindexl03[j]) && abs(bufferindexl03[j]) <= 1.0f) {
				//printf("writing:%.8f\n", bufferindexl03[j]);
				visualbuffer03->push_back(bufferindexl03[j]);
			}
			else
			{
				//visualbuffer03->push_back(0.0);
			}
		}
	}
	
	return  0;
}

int AudioVisualizer::getNumberOfFrames()
{
	return nframes;
}

int AudioVisualizer::freeMem()
{
	delete model;
	delete stftl;
	delete stftr;
	return 0;
}

