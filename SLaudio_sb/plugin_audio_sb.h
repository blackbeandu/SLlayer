#include <NvInfer.h>
#include <cassert>
#include <string>

void cuda_audio_sb(float* output, float* input, float* audio_0, float* audio_1, int N, int S, int E, cudaStream_t stream);

#define  AUDIO_SB_PLUGIN_NAME   "audio_sb"
#define  AUDIO_SB_PLUGIN_VERSION   "1.0.0"

enum AUDIO_SB_ENUM { FASTSPEECH=0 };
inline int get_audio_sb_mode(std::string& mode)
{
    if (mode == "fastspeech")
        return FASTSPEECH;

    return -1;
}
struct PAudioSb {
    // don't use pointer, string, STL(vector, list, ...), because those are not serialized simply.
    int mode;
};

class PAudioSbPlugin final : public nvinfer1::IPluginV2DynamicExt
{
private:
    PAudioSb mAudioSb;
    std::string mPluginNamespace;
public:
    PAudioSbPlugin(const PAudioSb& fc) :mAudioSb(fc) { }
    PAudioSbPlugin(void const* data, size_t length) { mAudioSb = *(PAudioSb*)data; }
    PAudioSbPlugin() = delete;
    ~PAudioSbPlugin() = default;
    int32_t initialize() noexcept { return 0; }
    void terminate() noexcept {}
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept { return sizeof(PAudioSb); }
    void serialize(void* buffer) const noexcept { *(PAudioSb*)buffer = mAudioSb; }
    const char* getPluginType() const noexcept { const char* type = AUDIO_SB_PLUGIN_NAME; return type; }
    const char* getPluginVersion() const noexcept { const char* ver = AUDIO_SB_PLUGIN_VERSION; return ver; }
    void destroy() noexcept { delete this; }
    void setPluginNamespace(const char* pluginNamespace) noexcept { mPluginNamespace = pluginNamespace; }
    const char* getPluginNamespace() const noexcept { return mPluginNamespace.c_str(); }
    void attachToContext( cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept {}
    void detachFromContext() noexcept {}
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept { }
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept { PAudioSbPlugin* plugin = new(std::nothrow) PAudioSbPlugin(mAudioSb); plugin->setPluginNamespace(mPluginNamespace.c_str());  return plugin;  }
    int32_t getNbOutputs() const noexcept;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept;
    bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept;
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept;
};

class PAudioSbPluginCreator : public nvinfer1::IPluginCreator
{
public:
    PAudioSbPluginCreator() = default;

    const char* getPluginName() const noexcept
    {
        return AUDIO_SB_PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept
    {
        return AUDIO_SB_PLUGIN_VERSION;
    }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept { return nullptr; }

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
    {
        PAudioSbPlugin* plugin = new PAudioSbPlugin(*(PAudioSb*)fc);
        mPluginName = name;
        return plugin;
    }
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
    {
        PAudioSbPlugin* plugin = new PAudioSbPlugin(serialData, serialLength);
        mPluginName = name;
        return plugin;
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept
    {
        mNamespace = pluginNamespace;
    }
    const char* getPluginNamespace() const noexcept
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    std::string mPluginName;
};