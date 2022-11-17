// AIAudioVisualizer.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <fstream>
#include <tensorflow/c/c_api.h>
#include <scope_guard.hpp>
#include <deque>

#include "cppflow/ops.h"
#include "cppflow/model.h"
#include "AudioFile.h"
#include "Gamma/DFT.h"
#include "Gamma/Types.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "imgui_ai.h"

using namespace xt;
// TODO: Reference additional headers your program requires here.
class AudioVisualizer
{
public:
	void init();
	int loadModel();
	int initSound(int samplesperchannel);
	int predict(float* samples, int numchannels, float* bufflstart00, float* buffrstart00, float* bufflstart01, float* buffrstart01, float* bufflstart02, float* buffrstart02, float* bufflstart03, float* buffrstart03, std::deque<float> * visualbuffer00, std::deque<float>* visualbuffer01, std::deque<float>* visualbuffer02, std::deque<float>* visualbuffer03);
	int getNumberOfFrames();
	int freeMem();

private:
	cppflow::model* model;

	gam::STFT * stftl;
	gam::STFT * stftr;

	xt::xarray<float> superarray; //all frames
	xt::xarray<float> arraytransposed;
	xt::xarray<float> phasetransposed;

	xt::xarray<double> stftpredicted;
	xt::xarray<double> softmasks;
	xt::xarray<double> sources;

	int numSamples;
	int nframes; //hop size + 2 because it's padded with 0s
	int currentFrame;
	float scale = 2.0000630488853100000;
};
