// AIAudioVisualizer.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <fstream>
#include <tensorflow/c/c_api.h>
#include <scope_guard.hpp>

#include "cppflow/ops.h"
#include "cppflow/model.h"
#include "AudioFile.h"
#include "Gamma/DFT.h"
#include "Gamma/Types.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

using namespace xt;
// TODO: Reference additional headers your program requires here.
class AudioVisualizer
{
public:
	void init();
	int loadModel();
	int initSound();
private:
	cppflow::model* model;
	
};
