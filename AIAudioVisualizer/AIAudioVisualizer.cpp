#include <AIAudioVisualizer.h>
#include <vk_engine.h>

int main(int argc, char* argv[])
{
	AudioVisualizer* audio = new AudioVisualizer();

	//------------tensorflow init------------------
	//audio->init();
	//audio->loadModel();
	//audio->initSound();
	//------------tensorflow init------------------

	VulkanEngine engine;

	engine.init();

	engine.run();

	engine.cleanup();

	return 0;
}

void AudioVisualizer::init()
{
	printf("Init TensorFlow C library version %s\n", TF_Version());
}

int AudioVisualizer::loadModel()
{
	model = new cppflow::model(std::string("model"));
	auto operations = model->get_operations();

	std::cout << "Load graph success" << std::endl;
	std::cout << "Checking operations..." << std::endl;
	for (int i = 0; i < operations.size(); i++)
	{
		std::cout << operations[i] << std::endl;
	}
	return 0;
}

int AudioVisualizer::initSound()
{
	AudioFile<float> audioFile;
	audioFile.load("short.wav");

	audioFile.printSummary();
	std::cout << "Load sound success" << std::endl;

	float* bufferl = new float[1025]; //1024 + 1?
	float* bufferr = new float[1025]; //1024 + 1?

	float scale = 2.0000630488853100000;

	size_t size = 1 * 1025 * 21 * 2; //final size of tensor input
	std::vector<float> _data(size, 0.0f);    // make room for tensor and initialize them to 0
	std::vector<int64_t> shape{ 1 , 1025, 21, 2 };


	gam::STFT stftl{
		2048,		// Window size
		512 ,		// Hop size; number of samples between transforms
		0,			// Pad size; number of zero-valued samples appended to window
		gam::WindowType::HANN,		// Window type: BARTLETT, BLACKMAN, BLACKMAN_HARRIS,
					//		HAMMING, HANN, WELCH, NYQUIST, or RECTANGLE
		gam::SpectralType::COMPLEX		// Format of frequency samples:
					//		COMPLEX, MAG_PHASE, or MAG_FREQ
	};

	gam::STFT stftr{
		2048,		// Window size
		512 ,		// Hop size; number of samples between transforms
		0,			// Pad size; number of zero-valued samples appended to window
		gam::WindowType::HANN,		// Window type: BARTLETT, BLACKMAN, BLACKMAN_HARRIS,
					//		HAMMING, HANN, WELCH, NYQUIST, or RECTANGLE
		gam::SpectralType::COMPLEX		// Format of frequency samples:
					//		COMPLEX, MAG_PHASE, or MAG_FREQ
	};

	int numSamples = audioFile.getNumSamplesPerChannel();
	int nframes = (int)numSamples / 512 + 2; //hop size + 2 because it's padded with 0s
	int currentFrame = 0;
	auto superarray = xt::xarray<float>::from_shape({ 1 , 1025, (unsigned long) nframes, 2 }); //all frames
	auto arraytransposed = xt::xarray<float>::from_shape({ 1025, (unsigned long)nframes, 2 });
	auto phasetransposed = xt::xarray<float>::from_shape({ 1025, (unsigned long)nframes, 2 });

	for (int i = 0; i < 1536; i++)
	{
		stftl(0.0);
		stftr(0.0);
	}

	for (int i = 0; i < 512; i++)
	{
		stftl(audioFile.samples[0][i]);
		stftr(audioFile.samples[1][i]);
	}

	for (int i = 512; i < numSamples; i++)
	{
		float currentSample = audioFile.samples[0][i];

		if (stftl(currentSample)) {
			// Loop through all the bins
			for (unsigned k = 0; k < stftl.numBins(); ++k) {
				superarray(0, k, currentFrame, 0) = log2(stftl.bin(k).mag() * scale  + 1.0);
				arraytransposed(k, currentFrame,0) = stftl.bin(k).mag() ;
				phasetransposed(k, currentFrame, 0) = stftl.bin(k).phase();
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

		currentSample = audioFile.samples[1][i];

		if (stftr(currentSample)) {

			// Loop through all the bins
			for (unsigned k = 0; k < stftr.numBins(); ++k) {
				superarray(0, k, currentFrame, 1) = log2(stftr.bin(k).mag() * scale + 1.0);
				arraytransposed(k, currentFrame, 1) = stftr.bin(k).mag();
				phasetransposed(k, currentFrame, 1) = stftr.bin(k).phase();
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

	xt::xarray<double> stftpredicted = xt::zeros<double>({ 1025, nframes, 2, 4 });
	double division = 1.0 / 7.0;//dumb rounding bug, integer division was before (1/7 => 0)

	for (auto&& i : xt::arange(10, nframes - 11, 3)) {
		auto frameview = xt::view(superarray, xt::all(), xt::all(), xt::range(i - 10, i + 11), xt::all());

		std::copy(frameview.cbegin(), frameview.cend(), _data.begin());
		cppflow::tensor input = cppflow::tensor( _data , { 1, 1025, 21, 2 });

		auto predicted = model->operator()({{"serving_default_input_1", input}}, { "StatefulPartitionedCall"}); //solo funciona asi

		std::vector<float> values = predicted[0].get_data<float>();

		std::vector<std::size_t> shapeout = {1025, 21, 2, 4 };
		auto xtensorpredicted = xt::adapt(values, shapeout);

		auto stftpredictedslice = xt::view(stftpredicted, xt::all(), xt::range(i - 10, i + 11), xt::all(), xt::all());

		stftpredictedslice = stftpredictedslice + division * xtensorpredicted;//dumb rounding bug
	}

	stftpredicted = xt::pow(2, stftpredicted) - 1;
	auto denmask = xt::sum(stftpredicted, { 3 });
	auto softmasks = xt::xarray<double>::from_shape({ 1025, (unsigned long)nframes, 2, 4 });
	auto sources = xt::xarray<double>::from_shape({ 1025, (unsigned long)nframes, 2 , 4});
	
	float eps = std::numeric_limits<float>::epsilon();

	for (int j = 0; j < 4; j++)
	{
		auto softmaskslice = xt::view(softmasks, xt::all(), xt::all(), xt::all(), j);
		auto sourcesslice = xt::view(sources, xt::all(), xt::all(), xt::all(), j);
		auto stftpredictedslice = xt::view(stftpredicted, xt::all(), xt::all(), xt::all(), j);

		softmaskslice = stftpredictedslice / (denmask + eps);
		sourcesslice = softmaskslice * arraytransposed;
	}

	xt::xarray<double> magnitudestftoutl = xt::view(sources, xt::all(), xt::all(),0, xt::all());
	xt::xarray<double> magnitudestftoutr = xt::view(sources, xt::all(), xt::all(), 1, xt::all());

	//-----mix step
	AudioFile<float>::AudioBuffer bufferout;
	bufferout.resize(2);
	bufferout[0].resize(nframes * 512 + 2048); //number of frames plus hop size
	bufferout[1].resize(nframes * 512 + 2048); //number of frames plus hop size


	float* bufferindexl = &bufferout[0][0];
	float* bufferindexr = &bufferout[1][0];

	stftl.inverseWindowing(false);
	stftr.inverseWindowing(false);

	for (int k = 0; k < nframes; k++)
	{
		for (int j = 0; j < stftl.numBins(); j++)
		{
			gam::Complex comp0;
			gam::Complex comp1;

			comp0.fromPolar(magnitudestftoutl(j, k, 3), phasetransposed(j, k, 0));
			comp1.fromPolar(magnitudestftoutr(j, k, 3), phasetransposed(j, k, 1));

			stftl.bin(j).set(comp0);
			stftr.bin(j).set(comp1);
		}

		stftl.inverse(bufferindexl); //save inverse into buffer
		stftr.inverse(bufferindexr); //save inverse into buffer

		bufferindexl += 512; //hop size
		bufferindexr += 512; //hop size
	}
	
	bool ok = audioFile.setAudioBuffer(bufferout);
	audioFile.save("voice.wav", AudioFileFormat::Wave);

	delete model;
	return 0;
}

