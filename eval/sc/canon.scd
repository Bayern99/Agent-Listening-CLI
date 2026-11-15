~p = PARAMS;
~out = OUTPATH;
{
var ratios = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2];
var motif = [0, 2, 4, 3, 4, 2, 6, 4];
var step = 0.5, reps = 4, tail = 2.5;
var dur = max(~p[\v2Delay], ~p[\v3Delay]) + (reps * motif.size * step) + tail;
var score = Score.new;
var defs = [
	SynthDef(\bell, { |out = 0, freq = 440, amp = 0.2, att = 0.01, rel = 1.2, pan = 0|
		var env = EnvGen.ar(Env.perc(att, rel, curve: -4), doneAction: 2);
		var sig = SinOsc.ar(freq) + (SinOsc.ar(freq * 2.01) * 0.3) + (SinOsc.ar(freq * 3.98) * 0.12);
		Out.ar(out, Pan2.ar(sig * env * amp, pan));
	}),
	SynthDef(\reed, { |out = 0, freq = 220, amp = 0.2, res = 0, pan = 0|
		var env = EnvGen.ar(Env.linen(0.05, 0.3, 0.15), doneAction: 2);
		var sig = LPF.ar(Saw.ar(freq * [1, 1.003]).sum * 0.5, freq * 6);
		sig = sig + (BPF.ar(sig, 1500, 0.03) * res * 6);
		Out.ar(out, Pan2.ar(sig * env * amp, pan));
	}),
	SynthDef(\pluck, { |out = 0, freq = 110, amp = 0.2, att = 0.003, pan = 0|
		var env = EnvGen.ar(Env.perc(att, 0.8), doneAction: 2);
		var sig = LPF.ar(Pulse.ar(freq, 0.3), freq * 5) * 0.8;
		Out.ar(out, Pan2.ar(sig * env * amp, pan));
	})
];
var voice = { |start, def, base, args|
	reps.do { |rep|
		motif.do { |deg, i|
			var t = start + (((rep * motif.size) + i) * step);
			score.add([t, [\s_new, def, -1, 0, 0, \freq, base * ratios[deg]] ++ args]);
		};
	};
};
defs.do { |d| score.add([0.0, [\d_recv, d.asBytes]]) };
voice.(0.1, \bell, 523.25, [\amp, ~p[\v1Amp], \att, ~p[\v1Attack], \pan, -0.4]);
voice.(0.1 + ~p[\v2Delay], \reed, 523.25 * 2 / 3, [\amp, ~p[\v2Amp], \res, ~p[\v2Res], \pan, 0.4]);
voice.(0.1 + ~p[\v3Delay], \pluck, 130.81 * (2 ** (~p[\v3Detune] / 1200)), [\amp, ~p[\v3Amp], \pan, 0]);
score.add([dur, [\c_set, 0, 0]]);
score.sort;
score.recordNRT(
	outputFilePath: ~out, sampleRate: 48000, headerFormat: "WAV", sampleFormat: "int24",
	options: ServerOptions.new.numOutputBusChannels_(2), duration: dur,
	action: { 0.exit }
);
}.value;
