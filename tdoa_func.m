% function taus = tdoa_func(samples)
function tdoa_func()
    % path = strcat('./data/mic1', idx, '/')
    % refPath = strcat('./data/mic2', idx, '/')
    samples = 'beamformed_50099-50373.wav';
    refSig = strcat('./data/mics/mic1/', samples);
    sig2 = strcat('./data/mics/mic2/', samples);
    sig3 = strcat('./data/mics/mic3/', samples);
    sig4 = strcat('./data/mics/mic4/', samples);
    sig5 = strcat('./data/mics/mic5/', samples);
    
    [sig2_y, sig2_fs] = audioread(sig2);
    [sig3_y, sig3_fs] = audioread(sig3);
    [sig4_y, sig4_fs] = audioread(sig4);
    [sig5_y, sig5_fs] = audioread(sig5);   
    [ref_y, ref_fs] = audioread(refSig);
    
    Fs = 16000;
    [taus, R, lags] = gccphat([sig2_y sig3_y sig4_y sig5_y], ref_y, Fs);
    taus * Fs
    
    plot(1000*lags,real(R(:,1)))
    xlabel('Lag Times (ms)')
    ylabel('Cross-correlation')
    axis([-15,10,-.5,1.1])
    hold on
    plot(1000*lags,real(R(:,2)))
    plot(1000*lags,real(R(:,3)))
    hold off

    