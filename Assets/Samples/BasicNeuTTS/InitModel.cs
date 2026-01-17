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
        yield return new WaitUntil(() => !string.IsNullOrEmpty(AndroidObbMount.AndroidObbMount.patchMountPoint));

        tts.neuttsModelPath = GetAbsolutePath(tts.neuttsModelPath);
        tts.neucodecDecoderModelPath = GetAbsolutePatchPath(tts.neucodecDecoderModelPath);

        tts.phonemizerModelPath = GetAbsolutePatchPath(tts.phonemizerModelPath);
        tts.phonemizerConfigPath = GetAbsolutePatchPath(tts.phonemizerConfigPath);
        tts.phonemizerDictPath = GetAbsolutePatchPath(tts.phonemizerDictPath);

        tts.refAudioPath = GetAbsolutePatchPath(tts.refAudioPath);
        tts.refTranscriptPath = GetAbsolutePatchPath(tts.refTranscriptPath);

        tts.InitModel();
    }

    string GetAbsolutePath(string filepath)
    {
        return Path.IsPathRooted(filepath) ? filepath : Path.Join(AndroidObbMount.AndroidObbMount.mountPoint, filepath);
    }

    string GetAbsolutePatchPath(string filepath)
    {
        return Path.IsPathRooted(filepath) ? filepath : Path.Join(AndroidObbMount.AndroidObbMount.patchMountPoint, filepath);
    }
}
