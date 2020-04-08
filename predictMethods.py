from mido import MidiFile, MidiTrack, Message
import os
import numpy as np
import random

# get the scale for whatever chord has been placed
def getScale(root, type):
	# musical info
	major = [0, 2, 4, 5, 7, 9, 11, 12]
	minor = [0, 2, 3, 5, 7, 9, 11, 12]
	scale = []

	upperRoot = root

	if root < 30:
		upperRoot += 48
	elif root > 50:
		upperRoot += 12
	else:
		upperRoot += 36

	if type == 'maj':
		for i in range(len(major)):
			note = (root + major[i])
			upperNote = (upperRoot + major[i])
			if note < 112:
				scale.append(note)
			if upperNote < 112:
				scale.append(upperNote)
		return scale
	elif type == 'min':
		for i in range(len(minor)):
			note = (root + minor[i])
			upperNote = (upperRoot + major[i])
			if note < 112:
				scale.append(note)
			if upperNote < 112:
				scale.append(upperNote)
		return scale
	else:
		return 'ERROR'


def samples_to_midi(sample, fname, ticks_per_sample, thresh=0.50):

	# info for debug purposes
	totalCount = 0
	maxVal = -1
	minVal = 99

	num_notes = 96
	samples_per_measure = 96

	mid = MidiFile()
	track = MidiTrack()

	ticks_per_beat = mid.ticks_per_beat
	ticks_per_measure = 4 * ticks_per_beat
	ticks_per_sample = ticks_per_measure / samples_per_measure
	abs_time = 0
	last_time = 0

	for measure in sample:

		firstNote = -1
		rootPlaced = False
		scale = []

		notesInMeasure = 0
		melodyNotes = 0

		for y in range(measure.shape[0]-1):

			notesPlaced = 0
			abs_time += ticks_per_sample

			for x in range(measure.shape[1]):

				note = int(x + (128 - num_notes)/2)

				# 'boost' mid-range scale notes to encourage melody
				score = measure[y][x]

				if notesPlaced < 5 and len(scale) > 0:
					if note in scale:
						if random.random() < 0.0065:
							score += 0.5

				# limit number of simultaneous notes so we don't get weird chords
				if np.logical_and(score >= thresh, notesPlaced < 4).any():
					delta_time = int(abs_time - last_time)
					track.append(Message('note_on', note=note, velocity=127, time=delta_time))

					# determine chord
					if rootPlaced == False:

						rootPlaced = True
						firstNote = note

						if random.random() < 0.6:
							scale = getScale(firstNote, 'maj')
						else:
							scale = getScale(firstNote, 'min')

					totalCount += 1

					last_time = abs_time

				if y > 2:
					if np.logical_and(measure[y-3][x] >= thresh, score < thresh).any():
						delta_time = int(abs_time - last_time)
						track.append(Message('note_off', note=note, velocity=127, time=delta_time))
						last_time = abs_time

	print('total notes placed: '+str(totalCount))
	mid.tracks.append(track)
	mid.save('Output/'+fname+'.mid')
	print('file saved: '+'Output/'+fname+'.mid')