using System.Collections;
using System.IO;
using UnityEngine;

public class InitModel : MonoBehaviour
{
    public NeuTTS.NeuTTS tts;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    IEnumerator Start()
    {
        yield return new WaitUntil(() => !string.IsNullOrEmpty(AndroidObbMount.AndroidObbMount.mountPoint));

        tts.neuttsModelPath = GetAbsolutePath(tts.neuttsModelPath);
        tts.neucodecDecoderModelPath = GetAbsolutePath(tts.neucodecDecoderModelPath);

        tts.phonemizerModelPath = GetAbsolutePath(tts.phonemizerModelPath);
        tts.phonemizerConfigPath = GetAbsolutePath(tts.phonemizerConfigPath);
        tts.phonemizerDictPath = GetAbsolutePath(tts.phonemizerDictPath);

        tts.refAudioPath = GetAbsolutePath(tts.refAudioPath);
        tts.refTranscriptPath = GetAbsolutePath(tts.refTranscriptPath);

        tts.InitModel();
    }

    string GetAbsolutePath(string filepath)
    {
        return Path.IsPathRooted(filepath) ? filepath : Path.Join(AndroidObbMount.AndroidObbMount.mountPoint, filepath);
    }
}
