# Separation — MossFormer2_SS_16K

Two streams, orthogonal: `corr(src0, src1) = +0.0003`. Identical RMS to two decimals (−36.94 dB) is a
global-normalisation coincidence, not duplicate output.

| region | src0 − mixture | src1 − mixture |
| --- | --- | --- |
| speech 11.62–13.20 | **+9.17 dB** | −16.57 dB |
| cough 1 | −10.53 dB | −1.95 dB |
| cough 2 | −8.05 dB | −0.95 dB |
| exhalation 1 | −0.59 dB | **−47.31 dB** |
| silence | −19.97 dB | −29.39 dB |

**src0 is speech plus exhalations; src1 is the coughs** — a partition matching the branch split this
design draws, from a model that knows nothing about it.

Spectrograms confirm and add: src1 is near-floor everywhere outside the two coughs and keeps their full
0–8 kHz vertical structure; src0 keeps the speech harmonics and both exhalation noise bands. The
mixture's broadband floor is gone from both streams (−20 to −29 dB over silence), so separation is
denoising as well as partitioning — a confound for any claim that it "helped". src1 retains faint
500–1000 Hz residue at 5.3–6.3 s despite −47 dB RMS there, so the exhalation is attenuated, not removed.

**It cannot be scored for benefit on this recording.** SQUIM over the speech region moves 0.950 → 0.960,
which is nothing, because the speech and the airway events **never overlap in time**. A +9.17 dB gain
against non-concurrent sources buys no intelligibility. Separation earns its place only where speech and
airway coincide, and this file cannot demonstrate that either way — hence optional and off by default.

For enhancement variants across SNR, see [`snr.md`](snr.md). Note that `MossFormerGAN_SE_16K` and
`MossFormer2_SE_48K` are *enhancement* models and their behaviour does not transfer to this separator,
which has never been run under the SNR sweep.
