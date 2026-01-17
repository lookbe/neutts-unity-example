using LlamaCpp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using UnityEngine;

namespace NeuTTS
{
    public class DecoderModel : BackgroundRunner
    {
        [Header("Model")]
        public string modelPath = string.Empty;

        // Define a delegate (or use Action<T>)
        public delegate void StatusChangedDelegate(ModelStatus status);
        public event StatusChangedDelegate OnStatusChanged;

        private ModelStatus _status = ModelStatus.Init;

        // Public getter, no public setter
        public ModelStatus status
        {
            get => _status;
            protected set
            {
                if (_status != value)
                {
                    _status = value;
                    OnStatusChanged?.Invoke(_status);
                }
            }
        }

        protected void PostStatus(ModelStatus newStatus)
        {
            unityContext?.Post(_ => status = newStatus, null);
        }

        async void OnDestroy()
        {
            await BackgroundStop();
            FreeModel();
        }

        // Define a delegate (or use Action<T>)
        public delegate void ResponseGeneratedDelegate(float[] response);
        public event ResponseGeneratedDelegate OnResponseGenerated;

        InferenceSession _session;

        public void InitModel()
        {
            if (string.IsNullOrEmpty(modelPath))
            {
                Debug.LogError("path not set");
                return;
            }

            if (_status != ModelStatus.Init)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Loading;
            RunBackground(RunInitModel);
        }

        void RunInitModel(CancellationToken cts)
        {
            try
            {
                Debug.Log($"Load model at {modelPath}");

                _session = new InferenceSession(modelPath);
                if (_session == null)
                {
                    throw new System.Exception("unable to load model");
                }

                //string warmupText = "<|speech_1|><|speech_2|><|speech_3|><|speech_4|><|speech_5|>";
                //DecodeInternal(warmupText, cts);

                Debug.Log("Load model done");
                PostStatus(ModelStatus.Ready);
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"An unexpected error occurred: {ex.Message}");

                FreeModel();
                PostStatus(ModelStatus.Init);
            }
        }

        private class DecodePayload : IBackgroundPayload
        {
            public string Codes;
        }

        public void Decode(string codes)
        {
            // harcoded value from onnx
            if (string.IsNullOrEmpty(codes))
            {
                Debug.LogError("wrong codes");
                return;
            }

            if (_session == null)
            {
                Debug.LogError("model not loaded");
                return;
            }

            if (status != ModelStatus.Ready && status != ModelStatus.Generate)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Generate;
            RunBackground(new DecodePayload() { Codes = codes }, RunDecode);
        }

        void RunDecode(DecodePayload payload, CancellationToken cts)
        {
            try
            {
                float[] byteArray = DecodeInternal(payload.Codes, cts);
                PostResponse(byteArray);
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"An unexpected error occurred during RunDecode: {ex.Message}");
                PostResponse(new float[0]);
            }
            finally
            {
                PostStatus(ModelStatus.Ready);
            }

        }

        float[] DecodeInternal(string codes, CancellationToken cts)
        {
            var matches = Regex.Matches(codes, @"<\|speech_(\d+)\|>");

            List<int> speechIds = new List<int>();
            foreach (Match match in matches)
            {
                if (int.TryParse(match.Groups[1].Value, out int id))
                {
                    speechIds.Add(id);
                }
            }

            if (speechIds.Count > 0)
            {
                int[] dimensions = new int[] { 1, 1, speechIds.Count };
                return DecodeCode(speechIds.ToArray(), dimensions, cts);
            }

            return Array.Empty<float>();
        }

        float[] DecodeCode(int[] codes, int[] dimensions, CancellationToken cts)
        {
            if (dimensions.Length != 3 || dimensions[1] != 1)
            {
                throw new ArgumentException("Codes should be of shape [B, 1, F].");
            }

            var inputTensor = new DenseTensor<int>(codes, dimensions);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("codes", inputTensor)
            };

            using (var results = _session.Run(inputs))
            {
                var output = results.First().AsTensor<float>();
                return output.ToArray();
            }
        }

        void FreeModel()
        {
            _session?.Dispose();
        }

        void PostResponse(float[] response)
        {
            unityContext?.Post(_ => OnResponseGenerated?.Invoke(response), null);
        }

        float[] LinearOverlapAdd(List<float[]> frames, int stride)
        {
            if (frames.Count == 0) return Array.Empty<float>();

            // Calculate total size
            int totalSize = 0;
            for (int i = 0; i < frames.Count; i++)
            {
                int frameEnd = stride * i + frames[i].Length;
                if (frameEnd > totalSize) totalSize = frameEnd;
            }

            float[] outBuf = new float[totalSize];
            float[] sumWeight = new float[totalSize];

            int offset = 0;
            foreach (var frame in frames)
            {
                int frameLength = frame.Length;
                for (int j = 0; j < frameLength; j++)
                {
                    // Linear weight: weight = np.abs(0.5 - (t - 0.5))
                    float t = (float)(j + 1) / (frameLength + 1);
                    float weight = Math.Abs(0.5f - (t - 0.5f));

                    outBuf[offset + j] += weight * frame[j];
                    sumWeight[offset + j] += weight;
                }
                offset += stride;
            }

            // Normalize by weights
            for (int i = 0; i < outBuf.Length; i++)
            {
                if (sumWeight[i] > 0) outBuf[i] /= sumWeight[i];
            }

            return outBuf;
        }
    }
}
