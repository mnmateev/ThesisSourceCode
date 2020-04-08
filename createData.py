from mido import MidiFile, MidiTrack, Message
import os
import numpy as np
import sys

num_notes = 96
samples_per_measure = 96

# turn midi file into data model can understand
def parseMidi(filename):
    print(filename)
    has_time_sig = False
    flag_warning = False
    mid = MidiFile(filename)
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_measure = 4 * ticks_per_beat

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'time_signature':
                new_tpm = msg.numerator * ticks_per_beat * 4 / msg.denominator
                if has_time_sig and new_tpm != ticks_per_measure:
                    flag_warning = True
                ticks_per_measure = new_tpm
                has_time_sig = True
    if flag_warning:
        print("  ^^^^^^ WARNING ^^^^^^")
        print("    " + filename)
        print("    Detected multiple distinct time signatures.")
        print("  ^^^^^^ WARNING ^^^^^^")
        return 'ERROR'

    all_notes = {}
    for i, track in enumerate(mid.tracks):
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on':
                note = msg.note - (128 - num_notes) / 2
                if(note >= 0 and note < num_notes):
                    if note not in all_notes:
                        all_notes[note] = []
                    else:
                        single_note = all_notes[note][-1]
                        if len(single_note) == 1:
                            single_note.append(single_note[0] + 1)
                    all_notes[note].append([abs_time * samples_per_measure / ticks_per_measure])
            elif msg.type == 'note_off':
                if len(all_notes[note][-1]) != 1:
                    continue
                all_notes[note][-1].append(abs_time * samples_per_measure / ticks_per_measure)

    for note in all_notes:
        for start_end in all_notes[note]:
            if len(start_end) == 1:
                start_end.append(start_end[0] + 1)

    samples = []

    for note in all_notes:
        for start, end in all_notes[note]:
            sample_ix = start / samples_per_measure

            while len(samples) <= sample_ix:
                samples.append(np.zeros((samples_per_measure, num_notes), dtype=np.int32))

            sample = samples[int(sample_ix)]
            start_ix = start - sample_ix * samples_per_measure

            sample[int(start_ix), int(note)] = 1

    return samples

def reshape(samples):

  numNotes = samples[0].shape[1]
  minNote, maxNote = transposeRange(samples)
  s = numNotes/2 - (maxNote + minNote)/2
  outSamples = samples
  outLens = [len(samples), len(samples)]

  for i in range(len(samples)):
    outSample = np.zeros_like(samples[i])
    outSample[:,int(minNote+s):int(maxNote+s)] = samples[i][:,int(minNote):int(maxNote)]
    outSamples.append(outSample)

  return outSamples, outLens

def transposeRange(samples):

  mergedSample = np.zeros_like(samples[0])

  for sample in samples:
    mergedSample = np.maximum(mergedSample, sample)

  mergedSample = np.amax(mergedSample, axis=0)
  minNote = np.argmax(mergedSample)
  maxNote = mergedSample.shape[0] - np.argmax(mergedSample[::-1])

  return minNote, maxNote

all_samples = []
all_lens = []
print("Loading Songs...")

folder = sys.argv[1]

for root, subdirs, files in os.walk('Samples/'+folder):
    for file in files:
        path = root + "/" + file

        samples = parseMidi(path)

        if type(samples) != str:
            samples, lens = reshape(samples)
            all_samples += samples
            all_lens += lens

print("Saving " + str(len(all_samples)) + " samples...")
print(str(all_lens) + ': all_lens')
all_samples = np.array(all_samples, dtype=np.int32)
print(all_samples.shape)
all_lens = np.array(all_lens, dtype=np.int32)
print(all_lens.shape)
np.save('Samples/'+folder+'.npy', all_samples)
print('Saved to '+'Samples/'+folder+'.npy')