#include <NvInfer.h>
#include <cinttypes>
#include "plugin_audio_sb.h"


#include <cstring>
using namespace std;


void reg()
{
	nvinfer1::IPluginRegistry* reg = getPluginRegistry();

	static PAudioSbPluginCreator pc;
	const nvinfer1::AsciiChar* name0 = pc.getPluginName();
	const nvinfer1::AsciiChar* version0 = pc.getPluginVersion();

	int numCreators;
	nvinfer1::IPluginCreator* const* ctors = reg->getPluginCreatorList(&numCreators);
	for (int idx = 0; idx < numCreators; idx++) {
		const nvinfer1::AsciiChar* name = ctors[idx]->getPluginName();
		const nvinfer1::AsciiChar* version = ctors[idx]->getPluginVersion();
		if (strcmp(name0, name) == 0 && strcmp(version0, version) == 0) {
			return;
		}
	}
	bool b = reg->registerCreator(pc, "");
}
void dereg()
{
	nvinfer1::IPluginRegistry* reg = getPluginRegistry();
	nvinfer1::IPluginCreator* creator = reg->getPluginCreator(AUDIO_SB_PLUGIN_NAME, AUDIO_SB_PLUGIN_VERSION);
	bool b = reg->deregisterCreator(*creator);
}

#ifdef WIN64
int module_ref_count = 0;
using HINSTANCE = struct HINSTANCE__ { int unused; }*;
int __stdcall DllMain(HINSTANCE hinstDLL, unsigned long fdwReason, void* lpReserved)
{
	module_ref_count += 1;
	if (fdwReason == 1) reg();
	else if (fdwReason == 0) dereg();
	return 1;
}
#else
void __attribute__((constructor)) start_module(void)
{
	reg();
}
void __attribute__((destructor)) end_module(void)
{
	dereg();
}
#endif
int32_t PAudioSbPlugin::getNbOutputs() const noexcept
{
	return 1;
}
nvinfer1::DimsExprs PAudioSbPlugin::getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
	nvinfer1::DimsExprs output(inputs[0]);
    //auto two = exprBuilder.constant(2);
    //output.d[1] = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *output.d[1], *two);
	return output;
}
size_t PAudioSbPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
	//nvinfer1::Dims input_dims = inputs[0].dims;
	return 0;
}
bool PAudioSbPlugin::supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    assert(nbInputs == 3);
    assert(nbOutputs == 1);

    const nvinfer1::PluginTensorDesc* input = &inOut[pos];
    bool b = (input->type == nvinfer1::DataType::kFLOAT) && (input->format == nvinfer1::TensorFormat::kLINEAR);
    return b;
}

nvinfer1::DataType PAudioSbPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept
{
    // one outputs
    assert(index == 0);

    //if (index == 0) {
    //    return inputTypes[0];
    //}
    return nvinfer1::DataType::kFLOAT;
}


int32_t PAudioSbPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
	const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
	if (mAudioSb.mode == FASTSPEECH) {
		float* input = (float*)inputs[0];
		float* audio_0 = (float*)inputs[1];
		float* audio_1 = (float*)inputs[2];
		float* output = (float*)outputs[0];
		int N = outputDesc->dims.d[0];
		int S = outputDesc->dims.d[1];
		int E = outputDesc->dims.d[2];
		cuda_audio_sb(output, input, audio_0, audio_1, N, S, E, stream);
	}
	return 0;
}