~p = PARAMS;
~out = OUTPATH;
{
var ratios = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2, 9/4, 5/2];
var theme = [0, 2, 4, 3, 4, 2, 6, 4];
var inverted = theme.collect { |d| 4 - (d - 4) }.collect { |d| d.clip(0, 9) };
var step = 0.5, dur = 140;
var score = Score.new;
var defs = [
	SynthDef(\bell, { |out = 0, freq = 440, amp = 0.2, pan = 0|
		var env = EnvGen.ar(Env.perc(0.01, 1.4, curve: -4), doneAction: 2);
		var sig = SinOsc.ar(freq) + (SinOsc.ar(freq * 2.01) * 0.3) + (SinOsc.ar(freq * 3.98) * 0.12);
		Out.ar(out, Pan2.ar(sig * env * amp, pan));
	}),
	SynthDef(\reed, { |out = 0, freq = 220, amp = 0.2, sus = 0.3, pan = 0|
		var env = EnvGen.ar(Env.linen(0.06, sus, 0.2), doneAction: 2);
		var sig = LPF.ar(Saw.ar(freq * [1, 1.003]).sum * 0.5, freq * 5);
		Out.ar(out, Pan2.ar(sig * env * amp, pan));
	}),
	SynthDef(\pluck, { |out = 0, freq = 110, amp = 0.2, pan = 0|
		var env = EnvGen.ar(Env.perc(0.004, 0.7), doneAction: 2);
		var sig = LPF.ar(Pulse.ar(freq, 0.3), freq * 5) * 0.8;
		Out.ar(out, Pan2.ar(sig * env * amp, pan));
	})
];
var phrase = { |start, def, base, degrees, amp, pan|
	degrees.do { |deg, i|
		score.add([start + (i * step), [\s_new, def, -1, 0, 0, \freq, base * ratios[deg], \amp, amp, \pan, pan]]);
	};
};
var drone = { |start, end, freq, amp|
	var t = start;
	while { t < end } {
		score.add([t, [\s_new, \reed, -1, 0, 0, \freq, freq, \amp, amp, \sus, 3.6, \pan, 0]]);
		t = t + 4;
	};
};
var ostinato = { |start, end, base, pattern, amp|
	var t = start, i = 0;
	while { t < end } {
		score.add([t, [\s_new, \pluck, -1, 0, 0, \freq, base * ratios[pattern.wrapAt(i)], \amp, amp, \pan, 0.3]]);
		t = t + 0.25; i = i + 1;
	};
};
defs.do { |d| score.add([0.0, [\d_recv, d.asBytes]]) };
phrase.(0.5, \bell, 523.25, theme, 0.22, -0.2);
drone.(0.5, 8, 130.81, 0.05);
drone.(8, 44, 196.0, 0.07);
ostinato.(8, 44, 261.63, [0, 4, 7, 4, 2, 4], 0.08);
phrase.(44.5, \bell, 523.25 * 4 / 3, theme, 0.22, -0.2);
drone.(52, 92, 146.83, 0.07);
ostinato.(52, 92, 196.0, [7, 5, 3, 1, 3, 5], 0.08);
if(~p[\returnAt92]) { phrase.(92.5, \reed, 523.25, inverted, 0.16, 0.2) };
drone.(100, 128, 174.61, 0.07);
ostinato.(100, 128, 220.0, [0, 1, 2, 4, 2, 1], 0.07);
phrase.(128.5, \bell, 523.25, theme, 0.22, -0.2);
drone.(128.5, 136, 130.81, 0.05);
score.add([dur, [\c_set, 0, 0]]);
score.sort;
score.recordNRT(
	outputFilePath: ~out, sampleRate: 48000, headerFormat: "WAV", sampleFormat: "int24",
	options: ServerOptions.new.numOutputBusChannels_(2), duration: dur,
	action: { 0.exit }
);
}.value;
