// Every number on this page lives here, on purpose. These are REAL results from
// this repo's pipeline (out/metrics.json + out/ablation.json), not invented — but
// they come from ONE downloaded audio chunk of ONE season, so the test set is
// tiny (n = 9 held-out hive-days). The page says so wherever a number appears.
// Re-running scripts/ablate.py overwrites these.

export const REAL = true
export const CAVEAT = 'It comes from one 2021 audio chunk: about 43 hours of sound, 3 hives, and 9 held-out test days.'

// The corpus, stated plainly and early.
export const DATASET = {
  hives: 10, // UrBAN total; this slice uses 3
  hivesUsed: 3,
  city: 'Montréal',
  years: '2021–2022',
  sliceYear: 2021,
  hoursUsed: 43.2,
  clips: 176,
  sampleRateHz: 16000,
  fobMeaning:
    'A frame of bees is one wooden frame inside the hive box. Beekeepers count how many frames are covered with bees at each inspection; that count is the colony-strength number the audio tries to predict.',
}

// The three hives actually used, with their real inspection FoB ranges.
export const HIVES = [
  { tag: '3629', clips: 61, fob: '6–15', note: 'a weaker colony' },
  { tag: '3690', clips: 59, fob: '13–30', note: 'a strong colony' },
  { tag: '3627', clips: 56, fob: '14–20', note: 'the hive I held out; its count actually changes over the season' },
]

// The hero contrast: one hive (3627), scored two ways on the SAME held-out days.
// same_hive = the model trained on 3627's own earlier days (it has seen this hive).
// loho      = the model trained ONLY on the other hives (it has never heard 3627).
// The gap between the two MAEs is the hive-identity leak.
// One consistent run (out/ablation.json, n = 9 held-out days for hive 3627).
export const HEADLINE = {
  testId: '3627',
  nTest: 9,
  sameHiveMae: 3.3,
  sameHiveCi: [2.57, 3.92] as [number, number],
  sameHiveMape: 22, // average prediction is 22% off the true count
  lohoMae: 11.55,
  lohoCi: [9.08, 13.32] as [number, number],
  lohoMape: 77,
  meanBaselineMae: 11.88, // a lazy model that always answers the training mean
  skillSamePct: 72, // same-hive error is 72% below the lazy baseline
  skillLohoPct: 3, // held-out error is only 3% below it: no real skill
  overfitGapPct: 250, // held-out error is 250% higher than same-hive
  fobUnit: 'frames',
}

// Slice 2: audio vs internal temperature vs both, on the SAME held-out days.
// The story is the columns, not any single cell. Temperature is the BEST predictor
// when the hive is in training and the WORST when it is held out — it tracks hive
// and season identity, not colony strength. That is a second leak, not free signal.
export const ABLATION: {
  feature: string
  label: string
  sameHive: number
  loho: number
}[] = [
  { feature: 'audio', label: 'Hive audio (log-mel)', sameHive: 3.3, loho: 11.55 },
  { feature: 'temp', label: 'Internal temp + humidity', sameHive: 2.68, loho: 13.1 },
  { feature: 'both', label: 'Audio + temp', sameHive: 3.37, loho: 10.87 },
]

// What the literature already established, so the page does not overclaim novelty.
export const PRIOR = {
  cite: 'UrBAN follow-up (Modulation Tensorgrams + CRDNN-3D, arXiv 2607.20386)',
  randomR: 0.76,
  hiveIndependentR: 0.5,
}

export const STACK = [
  {
    layer: 'Reading the audio',
    choice: "Python's built-in wave module + numpy",
    why: "The wave module opens a .wav file and hands me back the raw sound as a list of numbers. numpy holds that list so I can do math on it. The recordings are plain 16 kHz mono, so I never needed a heavier audio library.",
  },
  {
    layer: 'Turning sound into features',
    choice: 'scipy STFT + a numpy mel filterbank',
    why: 'The STFT slices a clip into short time windows and tells me how much energy sits at each pitch. The mel filterbank bunches those pitches into a smaller set of bands, roughly the way your ear does. I average that across the clip, so each recording becomes one short row of numbers.',
  },
  {
    layer: 'The model',
    choice: 'scikit-learn Ridge regression',
    why: 'Ridge takes that row of numbers and returns one number, the predicted frame count. It fits a straight line with a small penalty that stops it overreacting when there are only a handful of labels, which is my case exactly.',
  },
  {
    layer: 'Scoring it',
    choice: 'MAE, RMSE, Spearman, and a bootstrap',
    why: 'MAE and RMSE say how far off the guesses are. Spearman checks whether they at least put the hives in the right order. The bootstrap re-runs the scoring on thousands of resampled test sets to put an error range around each number, so I can see how shaky a small sample makes it.',
  },
  {
    layer: 'The two splits',
    choice: 'leave-one-hive-out vs same-hive',
    why: 'One split hides a whole hive from training and the other lets the model see it. Comparing the two is how I catch the model leaning on hive identity instead of sound.',
  },
  {
    layer: 'This page',
    choice: 'Vite, React 19, TypeScript, Tailwind 4',
    why: 'Vite builds the page, React draws it, TypeScript catches my mistakes as I write, and Tailwind handles the styling. Every number you see comes from one file, data.ts, so nothing is hand-typed into the page or fetched live.',
  },
]
