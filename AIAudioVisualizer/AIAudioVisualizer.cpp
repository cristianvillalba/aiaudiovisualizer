#include <AIAudioVisualizer.h>
#include <vk_engine.h>

int main(int argc, char* argv[])
{
	AudioVisualizer* audio = new AudioVisualizer();

	audio->init();
	audio->loadModel();
	audio->initSound();

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

	float scale = 1.0 / (pow(1024, 2));
	scale = sqrt(scale);
	scale = 1.0;

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
	int nframes = (int)numSamples / 512 + 2; //hop size
	int currentFrame = 0;
	auto superarray = xt::xarray<float>::from_shape({ 1 , 1025, (unsigned long) nframes, 2 }); //all frames
	auto arraytransposed = xt::xarray<float>::from_shape({ 1025, (unsigned long)nframes, 2 });

	for (int i = 0; i < numSamples; i++)
	{
		float currentSample = audioFile.samples[0][i];

		if (stftl(currentSample)) {

			// Loop through all the bins
			for (unsigned k = 0; k < stftl.numBins(); ++k) {
				// Here we simply scale the complex sample
				//std::cout << (stftl.bin(k).real() * scale) << "," << (stftl.bin(k).imag() * scale) << std::endl;
				//bufferl[k] = (stftl.bin(k).mag() + 1.0); //compute the absolute value;
				superarray(0, k, currentFrame, 0) = log2(stftl.bin(k).mag() + 1.0);

				arraytransposed(k, currentFrame,0) = log2(stftl.bin(k).mag() + 1.0);
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

			}
		}

		currentSample = audioFile.samples[1][i];

		if (stftr(currentSample)) {

			// Loop through all the bins
			for (unsigned k = 0; k < stftr.numBins(); ++k) {
				// Here we simply scale the complex sample
				//std::cout << (stftr.bin(k).real() * scale) << "," << (stftr.bin(k).imag() * scale) << std::endl;
				//bufferr[k] = log2(stftr.bin(k).mag() + 1.0); //compute the absolute value;
				superarray(0, k, currentFrame, 1) = log2(stftr.bin(k).mag() + 1.0);

				arraytransposed(k, currentFrame, 1) = log2(stftr.bin(k).mag() + 1.0);
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

	//------old xtensor code---------------
	//xt::xarray<float> xshape = xt::transpose(superarray, { 1, 2, 0 });
	//xshape.reshape({ 1, 1025, 1, 2});
	//xshape.reshape({ 1, 1025, 21, 2 });//with 21 frames
	
	cppflow::tensor input{ _data, shape };
	xt::xarray<float> stftpredicted = xt::empty<double>({ 1025, nframes, 2, 4 });

	for (auto&& i : xt::arange(10, nframes - 11, 3)) {
		//magnitudestftin[:,:,i-10:i+11,:]
		auto frameview = xt::view(superarray, xt::all(), xt::all(), xt::range(i - 10, i + 11), xt::all());

		std::copy(frameview.cbegin(), frameview.cend(), _data.begin());

		auto predicted = model->operator()({{"serving_default_input_1", input}}, { "StatefulPartitionedCall"}); //solo funciona asi
		//std::cout << predicted[0] << std::endl;

		std::vector<float> values = predicted[0].get_data<float>();
		std::vector<std::size_t> shape = { 1025, 21, 2, 4 };
		auto xtensorpredicted = xt::adapt(values, shape);

		//auto xtensorpredictedexpanded = xt::expand_dims(xtensorpredicted, 0);//esto no va
		//stftpredicted[:,i-10:i+11,:,:] = stftpredicted[:,i-10:i+11,:,:] + (1/7)*prediction[0,:,:,:,:] 
		auto stftpredictedslice = xt::view(stftpredicted, xt::all(), xt::range(i - 10, i + 11), xt::all(), xt::all());

		stftpredictedslice = stftpredictedslice + (1 / 7) * xtensorpredicted;

		//std::cout << predicted[0] << std::endl;

	}
	
	stftpredicted = xt::pow(2, stftpredicted) - 1;
	auto denmask = xt::sum(stftpredicted, { 3 });
	xt::xarray<float> softmasks = xt::zeros<float>({ 1025, nframes, 2, 4 });
	xt::xarray<float> sources = xt::zeros<float>({ 1025, nframes, 2, 4 });
	
	float eps = std::numeric_limits<float>::epsilon();

	
	for (int j = 0; j < 4; j++)
	{
		auto softmaskslice = xt::view(softmasks, xt::all(), xt::all(), xt::all(), j);
		auto sourcesslice = xt::view(sources, xt::all(), xt::all(), xt::all(), j);
		auto stftpredictedslice = xt::view(stftpredicted, xt::all(), xt::all(), xt::all(), j);

		softmaskslice = stftpredictedslice / (denmask + eps);
		sourcesslice = softmaskslice * arraytransposed;
	}
	
	//------testing tensor init------------
	//float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	//auto input = cppflow::fill({ 1, 1025, 21, 2 }, r ); //test tensor



	delete model;
	return 0;
}

