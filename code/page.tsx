"use client";
import React, { useState, useRef, useEffect, useCallback } from "react";
import { Camera, RefreshCw, CheckCircle, AlertCircle, Activity, Mic, Hand, MessageCircle, Eye, Volume2, Play, Square, MicOff } from "lucide-react";

const API_BASE_URL = "http://127.0.0.1:5000";

const useApiRequest = () => {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const makeRequest = useCallback(
        async (endpoint: string, method: string = "POST", body?: any) => {
            setIsLoading(true);
            setError(null);
            try {
                const config: RequestInit = {
                    method,
                    headers: { "Content-Type": "application/json" },
                };

                if (body && method !== "GET") {
                    config.body = JSON.stringify(body);
                }

                const res = await fetch(`${API_BASE_URL}${endpoint}`, config);
                if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
                return await res.json();
            } catch (err: any) {
                setError(err.message);
                throw err;
            } finally {
                setIsLoading(false);
            }
        },
        []
    );

    return { makeRequest, isLoading, error };
};

const useCamera = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isCameraOn, setIsCameraOn] = useState(false);
    const [cameraError, setCameraError] = useState<string | null>(null);

    const startCamera = async () => {
        try {
            setCameraError(null);
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 640,
                    height: 480,
                    facingMode: "user"
                }
            });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setIsCameraOn(true);
            }
        } catch (err: any) {
            console.error("Camera error:", err);
            setCameraError("Unable to access camera. Please ensure camera permissions are granted.");
            setIsCameraOn(false);
        }
    };

    const stopCamera = () => {
        const stream = videoRef.current?.srcObject as MediaStream;
        stream?.getTracks().forEach((t) => t.stop());
        if (videoRef.current) videoRef.current.srcObject = null;
        setIsCameraOn(false);
        setCameraError(null);
    };

    const captureFrame = () => {
        if (!videoRef.current || !canvasRef.current) return null;
        const ctx = canvasRef.current.getContext("2d");
        if (!ctx) return null;

        const video = videoRef.current;
        canvasRef.current.width = video.videoWidth;
        canvasRef.current.height = video.videoHeight;

        ctx.drawImage(video, 0, 0);
        return canvasRef.current.toDataURL("image/jpeg", 0.8).split(",")[1];
    };

    return { videoRef, canvasRef, isCameraOn, startCamera, stopCamera, captureFrame, cameraError };
};

interface StepData {
    step: number;
    type: string;
    instruction: string;
    direction?: string;
    challenge?: string;
    gesture?: string;
    completed: boolean;
}

// Function to convert audio Blob to WAV format
const convertBlobToWav = (blob: Blob) => {
    return new Promise<string>((resolve, reject) => {
        const audioContext = new (window.AudioContext)();
        const reader = new FileReader();

        reader.onloadend = async () => {
            try {
                const arrayBuffer = reader.result as ArrayBuffer;
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                const numChannels = 1;
                const sampleRate = audioContext.sampleRate;
                const numSamples = audioBuffer.length;
                const wavHeader = new ArrayBuffer(44);
                const view = new DataView(wavHeader);
                const format = 1; // PCM
                const byteRate = sampleRate * numChannels * 2;
                const blockAlign = numChannels * 2;

                // RIFF chunk descriptor
                view.setUint32(0, 0x52494646, false); // "RIFF"
                view.setUint32(4, 36 + numSamples * 2, true); // file size
                view.setUint32(8, 0x57415645, false); // "WAVE"
                // fmt sub-chunk
                view.setUint32(12, 0x666d7420, false); // "fmt "
                view.setUint32(16, 16, true); // sub-chunk size
                view.setUint16(20, format, true); // audio format
                view.setUint16(22, numChannels, true); // number of channels
                view.setUint32(24, sampleRate, true); // sample rate
                view.setUint32(28, byteRate, true); // byte rate
                view.setUint16(32, blockAlign, true); // block align
                view.setUint16(34, 16, true); // bits per sample
                // data sub-chunk
                view.setUint32(36, 0x64617461, false); // "data"
                view.setUint32(40, numSamples * 2, true); // sub-chunk size

                const dataView = new DataView(new ArrayBuffer(numSamples * 2));
                const floatData = audioBuffer.getChannelData(0);

                for (let i = 0; i < numSamples; i++) {
                    const s = Math.max(-1, Math.min(1, floatData[i]));
                    dataView.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
                }

                const wavBlob = new Blob([wavHeader, dataView], { type: 'audio/wav' });
                const finalReader = new FileReader();
                finalReader.onloadend = () => {
                    resolve((finalReader.result as string).split(',')[1]);
                };
                finalReader.onerror = () => reject(new Error("Failed to read WAV blob"));
                finalReader.readAsDataURL(wavBlob);

            } catch (err) {
                reject(err);
            }
        };

        reader.onerror = () => reject(new Error("Failed to read original audio blob"));
        reader.readAsArrayBuffer(blob);
    });
};

const AdvancedLivenessApp: React.FC = () => {
    const { videoRef, canvasRef, isCameraOn, startCamera, stopCamera, captureFrame, cameraError } = useCamera();
    const api = useApiRequest();

    const [sessionId, setSessionId] = useState<string | null>(null);
    const [instruction, setInstruction] = useState<string | null>(null);
    const [isVerified, setIsVerified] = useState(false);
    const [currentStep, setCurrentStep] = useState(0);
    const [totalSteps, setTotalSteps] = useState(0);
    const [status, setStatus] = useState<string>("idle");
    const [stepType, setStepType] = useState<string>("");
    const [stepData, setStepData] = useState<StepData | null>(null);
    const [allSteps, setAllSteps] = useState<StepData[]>([]);
    const [voiceChallenge, setVoiceChallenge] = useState<string>("");
    const [timeoutMessage, setTimeoutMessage] = useState<string>("");
    const [micError, setMicError] = useState<string | null>(null);

    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);

    const handleVoiceRecording = async () => {
        if (!isRecording) {
            setMicError(null);
            setInstruction("Recording... Please say the phrase now.");
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                mediaRecorderRef.current = mediaRecorder;
                audioChunksRef.current = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunksRef.current.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    stream.getTracks().forEach(track => track.stop());
                    setInstruction("Processing audio...");
                    try {
                        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                        const audioData = await convertBlobToWav(audioBlob);

                        if (audioData && sessionId) {
                            const res = await api.makeRequest("/verify_speech", "POST", {
                                sessionId,
                                audioData,
                            });

                            if (res.success) {
                                if (res.message.includes("All verification completed")) {
                                    setIsVerified(true);
                                    setStatus("verified");
                                    setInstruction(res.message);
                                    setTimeout(stopLivenessDetection, 3000);
                                } else {
                                    setStatus("step_completed");
                                    setInstruction(res.message);
                                    setTimeout(async () => {
                                        const stepInfo = await api.makeRequest("/get_step_info", "POST", { sessionId });
                                        setCurrentStep(stepInfo.current_step);
                                        setStepType(stepInfo.step_data.type);
                                        setStepData(stepInfo.step_data);
                                        if (stepInfo.step_data.challenge) {
                                            setVoiceChallenge(stepInfo.step_data.challenge);
                                        }
                                        setStatus("in_progress");
                                    }, 1000);
                                }
                            } else {
                                setInstruction(res.message);
                            }
                        }
                    } catch (err: any) {
                        console.error("Audio processing/API failed:", err);
                        setInstruction("Speech verification failed. Please try again.");
                    }
                };

                mediaRecorder.start();
                setIsRecording(true);
            } catch (err: any) {
                console.error("Microphone access failed:", err);
                setMicError("Microphone access failed. Please check permissions.");
                setIsRecording(false);
            }
        } else {
            // Stop recording
            if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                mediaRecorderRef.current.stop();
                setIsRecording(false);
            }
        }
    };

    const startLivenessDetection = async () => {
        try {
            setIsVerified(false);
            setStatus("starting");
            setTimeoutMessage("");
            setMicError(null);

            await startCamera();

            const res = await api.makeRequest("/start_liveness", "POST");
            setSessionId(res.session_id);
            setInstruction(res.instruction);
            setCurrentStep(res.current_step);
            setTotalSteps(res.total_steps);
            setStepType(res.step_type);
            setStepData(res.step_data);
            setAllSteps(res.all_steps);
            setStatus("in_progress");

            if (res.step_data && res.step_data.challenge) {
                setVoiceChallenge(res.step_data.challenge);
            }
        } catch (err) {
            console.error("Liveness start failed:", err);
            setStatus("error");
            setInstruction("Failed to start liveness detection. Please try again.");
        }
    };

    const stopLivenessDetection = () => {
        stopCamera();
        if (isRecording) {
            mediaRecorderRef.current?.stop();
            setIsRecording(false);
        }
        setSessionId(null);
        setInstruction(null);
        setIsVerified(false);
        setCurrentStep(0);
        setTotalSteps(0);
        setStatus("idle");
        setStepType("");
        setStepData(null);
        setAllSteps([]);
        setVoiceChallenge("");
        setTimeoutMessage("");
        setMicError(null);
    };

    const resetSession = () => {
        stopLivenessDetection();
        setIsVerified(false);
        setStatus("idle");
    };

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (sessionId && isCameraOn && status === "in_progress" && stepType !== "voice") {
            interval = setInterval(async () => {
                const frame = captureFrame();
                if (frame) {
                    try {
                        const res = await api.makeRequest("/process_frame", "POST", {
                            sessionId,
                            imageData: frame,
                        });

                        setInstruction(res.instruction);
                        setCurrentStep(res.current_step || currentStep);
                        setTotalSteps(res.total_steps || totalSteps);
                        setStepType(res.step_type || stepType);

                        if (res.step_data) {
                            setStepData(res.step_data);
                            if (res.step_data.challenge) {
                                setVoiceChallenge(res.step_data.challenge);
                            }
                        }

                        if (res.status === "verified") {
                            setIsVerified(true);
                            setStatus("verified");
                            setTimeout(() => {
                                stopLivenessDetection();
                            }, 3000);
                        } else if (res.status === "no_face") {
                            setStatus("no_face");
                        } else if (res.status === "step_completed") {
                            setStatus("step_completed");
                            setTimeout(() => setStatus("in_progress"), 1000);
                        } else if (res.status === "timeout") {
                            setStatus("timeout");
                            setTimeoutMessage("Step timeout. Please try again.");
                        } else {
                            setStatus("in_progress");
                        }
                    } catch (err) {
                        console.error("Frame process failed:", err);
                        setStatus("error");
                        setInstruction("Error processing frame. Please try again.");
                    }
                }
            }, 500);
        }
        return () => clearInterval(interval);
    }, [sessionId, isCameraOn, status, stepType, captureFrame, api, currentStep, totalSteps, stopLivenessDetection]);

    const getStatusColor = () => {
        switch (status) {
            case "verified": return "text-green-400";
            case "error": case "no_face": case "timeout": return "text-red-400";
            case "in_progress": case "step_completed": return "text-blue-400";
            case "starting": return "text-yellow-400";
            default: return "text-gray-400";
        }
    };

    const getProgressPercentage = () => {
        if (totalSteps === 0) return 0;
        return Math.min(((currentStep - 1) / totalSteps) * 100, 100);
    };

    const getStepIcon = (type: string) => {
        switch (type) {
            case "movement": return <Activity className="w-5 h-5" />;
            case "voice": return <MessageCircle className="w-5 h-5" />;
            case "gesture": return <Hand className="w-5 h-5" />;
            default: return <CheckCircle className="w-5 h-5" />;
        }
    };

    const getStepColor = (step: number) => {
        if (step < currentStep) return "bg-green-500";
        if (step === currentStep) return "bg-blue-500";
        return "bg-gray-600";
    };

    const getDirectionArrow = (direction?: string) => {
        switch (direction) {
            case "LEFT": return "←";
            case "RIGHT": return "→";
            case "UP": return "↑";
            case "DOWN": return "↓";
            default: return "";
        }
    };

    const getGestureEmoji = (gesture?: string) => {
        switch (gesture) {
            case "thumbs_up": return "👍";
            case "peace_sign": return "✌️";
            case "open_palm": return "✋";
            default: return "👋";
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center p-6">
            <div className="bg-gray-800/90 backdrop-blur-sm p-8 rounded-2xl shadow-2xl w-full max-w-5xl border border-gray-700">
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
                        <Eye className="text-blue-400" />
                        Advanced Liveness Verification
                    </h1>
                    <p className="text-gray-300">Complete 3 verification steps to prove you're a real person</p>
                </div>

                {/* Step Progress Indicator */}
                {totalSteps > 0 && (
                    <div className="mb-6">
                        <div className="flex justify-between items-center mb-4">
                            {allSteps.map((step, index) => (
                                <div key={index} className="flex flex-col items-center">
                                    <div className={`w-12 h-12 rounded-full flex items-center justify-center text-white font-bold ${getStepColor(index + 1)} transition-all duration-300 shadow-lg`}>
                                        {index + 1 < currentStep ? (
                                            <CheckCircle className="w-6 h-6" />
                                        ) : (
                                            getStepIcon(step.type)
                                        )}
                                    </div>
                                    <span className="text-sm text-gray-300 mt-2 text-center max-w-20">
                                        {step.type === "movement" ? "Move Face" :
                                         step.type === "voice" ? "Voice" :
                                         step.type === "gesture" ? "Gesture" : "Step"}
                                    </span>
                                </div>
                            ))}
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                            <div
                                className="bg-gradient-to-r from-blue-500 to-green-500 h-3 rounded-full transition-all duration-500 shadow-inner"
                                style={{ width: `${getProgressPercentage()}%` }}
                            ></div>
                        </div>
                        <div className="text-center mt-2 text-sm text-gray-400">
                            Step {currentStep} of {totalSteps}
                        </div>
                    </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Camera Section */}
                    <div className="space-y-4">
                        <div className="bg-black rounded-xl overflow-hidden relative shadow-2xl">
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                muted
                                className="w-full h-80 object-cover"
                                style={{ transform: "scaleX(-1)" }}
                            />
                            <canvas ref={canvasRef} className="hidden" />

                            {/* Overlay for current step indication */}
                            {stepData && (
                                <div className="absolute top-4 left-4 right-4">
                                    <div className="bg-black/70 backdrop-blur-sm rounded-lg p-3 text-white text-center">
                                        {stepType === "movement" && (
                                            <div className="text-2xl mb-2">
                                                Move your head {getDirectionArrow(stepData.direction)}
                                                <span className="text-3xl ml-2">{getDirectionArrow(stepData.direction)}</span>
                                            </div>
                                        )}
                                        {stepType === "gesture" && (
                                            <div className="text-2xl mb-2">
                                                Show: <span className="text-4xl ml-2">{getGestureEmoji(stepData.gesture)}</span>
                                            </div>
                                        )}
                                        {/* {stepType === "voice" && voiceChallenge && (
                                            <div className="text-lg">
                                                Say: <span className="font-bold text-yellow-300">"{voiceChallenge}"</span>
                                            </div>
                                        )} */}
                                        {stepType === "voice" && voiceChallenge && (
                                            <div className="text-lg">
                                                Say: <span className="font-bold text-yellow-300">"{voiceChallenge}"</span>
                                            </div>
                                        )}
                                        {stepType === "voice" && !voiceChallenge && (
                                            <div className="text-lg text-red-400">
                                                Error: No challenge phrase received. Please restart session.
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Status indicator */}
                            <div className="absolute bottom-4 right-4">
                                <div className={`w-4 h-4 rounded-full ${
                                    isCameraOn ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                                }`}></div>
                            </div>
                        </div>

                        {/* Camera Controls */}
                        <div className="flex justify-center space-x-4">
                            {!sessionId ? (
                                <button
                                    onClick={startLivenessDetection}
                                    disabled={api.isLoading}
                                    className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-3 rounded-xl font-semibold transition-all duration-200 flex items-center gap-2 shadow-lg disabled:opacity-50"
                                >
                                    {api.isLoading ? (
                                        <RefreshCw className="w-5 h-5 animate-spin" />
                                    ) : (
                                        <Camera className="w-5 h-5" />
                                    )}
                                    Start Verification
                                </button>
                            ) : (
                                <div className="flex space-x-4">
                                    {stepType === "voice" && (
                                        <button
                                            onClick={handleVoiceRecording}
                                            className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 flex items-center gap-2 shadow-lg ${
                                                isRecording
                                                    ? 'bg-red-600 hover:bg-red-700 text-white animate-pulse'
                                                    : 'bg-green-600 hover:bg-green-700 text-white'
                                            }`}
                                        >
                                            {isRecording ? (
                                                <>
                                                    <Square className="w-5 h-5" />
                                                    Stop Recording
                                                </>
                                            ) : (
                                                <>
                                                    <Mic className="w-5 h-5" />
                                                    Start Recording
                                                </>
                                            )}
                                        </button>
                                    )}

                                    <button
                                        onClick={stopLivenessDetection}
                                        className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-xl font-semibold transition-all duration-200 flex items-center gap-2 shadow-lg"
                                    >
                                        <Square className="w-5 h-5" />
                                        Stop
                                    </button>

                                    <button
                                        onClick={resetSession}
                                        className="bg-gray-600 hover:bg-gray-700 text-white px-6 py-3 rounded-xl font-semibold transition-all duration-200 flex items-center gap-2 shadow-lg"
                                    >
                                        <RefreshCw className="w-5 h-5" />
                                        Reset
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Instructions and Status Section */}
                    <div className="space-y-6">
                        {/* Current Instruction */}
                        <div className="bg-gray-700/50 backdrop-blur-sm p-6 rounded-xl border border-gray-600">
                            <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                                <Volume2 className="text-blue-400" />
                                Instructions
                            </h3>
                            <p className={`text-lg font-medium ${getStatusColor()}`}>
                                {instruction || "Click 'Start Verification' to begin the liveness detection process"}
                            </p>
                            {timeoutMessage && (
                                <div className="mt-3 p-3 bg-red-600/20 border border-red-500 rounded-lg">
                                    <p className="text-red-400 flex items-center gap-2">
                                        <AlertCircle className="w-5 h-5" />
                                        {timeoutMessage}
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Current Step Details */}
                        {stepData && (
                            <div className="bg-gray-700/50 backdrop-blur-sm p-6 rounded-xl border border-gray-600">
                                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                                    {getStepIcon(stepData.type)}
                                    Current Step: {stepData.type.charAt(0).toUpperCase() + stepData.type.slice(1)}
                                </h3>

                                {stepType === "movement" && (
                                    <div className="text-center">
                                        <div className="text-6xl mb-4">{getDirectionArrow(stepData.direction)}</div>
                                        <p className="text-gray-300">Move your head in the direction shown above</p>
                                    </div>
                                )}

                                {stepType === "voice" && voiceChallenge && (
                                    <div className="text-center">
                                        <div className="bg-yellow-500/20 border border-yellow-400 rounded-lg p-4 mb-4">
                                            <p className="text-yellow-300 text-2xl font-bold">"{voiceChallenge}"</p>
                                        </div>
                                        <p className="text-gray-300">Click the microphone button and say the phrase above clearly</p>
                                    </div>
                                )}

                                {stepType === "gesture" && (
                                    <div className="text-center">
                                        <div className="text-8xl mb-4">{getGestureEmoji(stepData.gesture)}</div>
                                        <p className="text-gray-300">{stepData.instruction}</p>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Status Indicators */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <div className="bg-gray-700/50 backdrop-blur-sm p-4 rounded-xl border border-gray-600">
                                <div className="flex items-center gap-3">
                                    <Camera className={isCameraOn ? "text-green-400" : "text-red-400"} />
                                    <div>
                                        <p className="text-white font-medium">Camera</p>
                                        <p className={`text-sm ${isCameraOn ? "text-green-400" : "text-red-400"}`}>
                                            {isCameraOn ? "Active" : "Inactive"}
                                        </p>
                                    </div>
                                </div>
                                {cameraError && (
                                    <p className="text-red-400 text-sm mt-2">{cameraError}</p>
                                )}
                            </div>

                            <div className="bg-gray-700/50 backdrop-blur-sm p-4 rounded-xl border border-gray-600">
                                <div className="flex items-center gap-3">
                                    <Activity className={getStatusColor().replace('text-', 'text-')} />
                                    <div>
                                        <p className="text-white font-medium">Status</p>
                                        <p className={`text-sm ${getStatusColor()}`}>
                                            {status === "idle" && "Ready to start"}
                                            {status === "starting" && "Initializing..."}
                                            {status === "in_progress" && "In progress"}
                                            {status === "step_completed" && "Step completed!"}
                                            {status === "verified" && "Verified!"}
                                            {status === "no_face" && "No face detected"}
                                            {status === "error" && "Error occurred"}
                                            {status === "timeout" && "Step timed out"}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Verification Success */}
                        {isVerified && (
                            <div className="bg-green-600/20 border border-green-400 rounded-xl p-6 text-center">
                                <CheckCircle className="w-16 h-16 text-green-400 mx-auto mb-4" />
                                <h3 className="text-2xl font-bold text-green-400 mb-2">Verification Successful!</h3>
                                <p className="text-green-300">You have been verified as a real person.</p>
                            </div>
                        )}

                        {/* Error Messages */}
                        {api.error && (
                            <div className="bg-red-600/20 border border-red-400 rounded-xl p-4">
                                <div className="flex items-center gap-2 text-red-400">
                                    <AlertCircle className="w-5 h-5" />
                                    <span className="font-medium">Error: {api.error}</span>
                                </div>
                            </div>
                        )}
                        {micError && (
                            <div className="bg-red-600/20 border border-red-400 rounded-xl p-4">
                                <div className="flex items-center gap-2 text-red-400">
                                    <MicOff className="w-5 h-5" />
                                    <span className="font-medium">Microphone Error: {micError}</span>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="mt-8 text-center text-gray-400 text-sm">
                    <p>This advanced liveness detection uses computer vision, voice recognition, and gesture detection to verify human presence.</p>
                </div>
            </div>
        </div>
    );
};

export default AdvancedLivenessApp;
