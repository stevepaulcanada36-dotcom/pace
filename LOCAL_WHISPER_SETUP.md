# Free local voice transcription for PaceSense

PaceSense now defaults to local Whisper transcription, so voice recordings do not need to be sent to OpenRouter for transcription and do not incur per-recording API charges.

## Install on Windows PowerShell

From the PaceSense project folder, activate your virtual environment if you use one, then run:

```powershell
python -m pip install -e ".[dashboard,local-stt]"
```

Start the app as usual:

```powershell
pacesense dashboard --data data\races.csv
```

Open **Settings → AI models and voice input** and make sure **Speech-to-text method** is set to **Local Whisper — no per-recording API charge**.

## First run

The default local model is `base`. The first transcription downloads the model files, so the first run needs internet access and may take a little while. Later transcriptions run on your computer. The local model is loaded and reused while the app process remains running.

If your computer has enough memory and you want to try a more accurate model, set this environment variable before starting PaceSense:

```powershell
$env:PACESENSE_LOCAL_WHISPER_MODEL="small"
pacesense dashboard --data data\races.csv
```

For a lighter model, use `tiny`. The trade-off is that smaller models can make more transcription mistakes. Local inference uses CPU `int8` by default; optional device settings are `PACESENSE_LOCAL_WHISPER_DEVICE` and `PACESENSE_LOCAL_WHISPER_COMPUTE_TYPE`.

## What remains online?

Local Whisper only transcribes audio. If you enter an OpenRouter API key, PaceSense can still use your chosen models for chat, filling fields, and suggesting a refined version. Those LLM features may have their own rate limits or charges. Without a key, PaceSense uses its basic offline rules for those text tasks.

To switch back to hosted transcription, choose **OpenRouter — may cost money** under the same Settings section. OpenRouter audio requests can require a minimum balance even when other selected models are free.

## Privacy and limits

In local mode, the audio file is processed on your computer after the model is downloaded. PaceSense temporarily writes the recording to a temporary file for decoding and removes that file after transcription. Recordings above 25 MB are rejected. Local transcription has no per-request API charge, but it uses your computer's CPU/RAM and initial model download bandwidth.
