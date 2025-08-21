% sweep_snr_deg_rel2input.m
% 目标：让输出 y 相对原始输入 x 的 SNR 衰减为 {5,10,13,15} dB
% 定义：SNR_deg(dB) = 10*log10( sum|x|^2 / sum|y - x|^2 )
% 实现：y = chan(x) + n；其中 |y-x|^2 = |chan(x)-x|^2 + |n|^2（期望意义下）
%       先计算信道失真能量，再用噪声能量补齐目标“误差能量”。

clear; clc;

%% =============== 0) 顶层配置 ===============
base_input_folder  = 'G:\dataset\0p05\mat_slice\RF1';
base_output_folder = 'G:\dataset\0p05\sweepSNRpng\RF1';
subfolder_list     = [10,12,13,14,15,18];

Fs = 100e6;        % 采样率（Hz），用于 spectrogram
seg_ms = 50;       % 片段时长（仅说明；以文件内容为准）

% —— 目标“相对输入的 SNR 衰减值”（dB）——
snr_deg_list = [5, 10, 13, 15];   % ΔdB：输出相对输入的 SNR 衰减

% —— 信道选项：'none' | 'rayleigh' | 'rician' ——（两者“共同造成”衰减）
fading_type = 'rayleigh';
fd_hz       = 30;     % 最大多普勒 (Hz)；50 ms 片段内约 3 次起伏
K_dB        = 6;      % Rician K 因子（dB）
K_linear    = 10^(K_dB/10);   % ★ Rician 需要线性 K

% —— STFT 参数（与主文统一） ——
win_len  = 2048;
overlap  = win_len/2;
nfft     = 2048;
win_func = hamming(win_len);

% —— 输出时频图尺寸 ——
desired_rows = 512; desired_cols = 512;

% —— 可复现 ——
base_seed = 20250818; rng(base_seed);

%% =============== 1) 批处理 ===============
for sf_idx = 1:numel(subfolder_list)
    sub_name  = num2str(subfolder_list(sf_idx));
    in_dir    = fullfile(base_input_folder,  sub_name);
    out_dir   = fullfile(base_output_folder, sub_name);
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    files = dir(fullfile(in_dir, '*.mat'));
    if isempty(files)
        fprintf('[警告] "%s" 无 .mat 文件，跳过。\n', in_dir);
        continue;
    end

    % 记录表：目标 ΔdB、达成 ΔdB、信道失真能量占比
    log_csv = fullfile(out_dir, sprintf('snr_deg_rel2input_log_%s.csv', sub_name));
    fid = fopen(log_csv, 'w');
    fprintf(fid, 'file,deg_target_db,deg_eff_db,chan_err_frac,noise_power_per_sample\n');

    fprintf('===============================\n处理子文件夹: %s  (共 %d 个文件)\n\n', in_dir, numel(files));

    for idx = 1:numel(files)
        mat_file = files(idx);
        mat_path = fullfile(mat_file.folder, mat_file.name);
        [~, stem, ~] = fileparts(mat_file.name);
        fprintf('(%d/%d) %s\n', idx, numel(files), mat_file.name);

        % ---------- 读取复数片段 ----------
        S = load(mat_path);
        if isfield(S,'RF1_I_fragment') && isfield(S,'RF1_Q_fragment')
            x = double(S.RF1_I_fragment) + 1j*double(S.RF1_Q_fragment);
        elseif isfield(S,'RF0_I_fragment') && isfield(S,'RF0_Q_fragment')
            x = double(S.RF0_I_fragment) + 1j*double(S.RF0_Q_fragment);
        else
            warning('文件缺少 RF*_I/Q_fragment 字段，跳过：%s', mat_file.name);
            continue;
        end
        x = x(:);                  % 保证列向量
        N = numel(x);
        P_sig = sum(abs(x).^2);    % 原始输入信号能量（参考能量）

        % ---------- 构造“可复现”的信道 ----------
        file_seed = base_seed + 1000*str2double(sub_name) + idx;
        switch lower(fading_type)
            case 'none'
                x_fad = x;
            case 'rayleigh'
                chan = comm.RayleighChannel( ...
                    'SampleRate', Fs, ...
                    'PathDelays', 0, ...
                    'AveragePathGains', 0, ...
                    'MaximumDopplerShift', fd_hz, ...
                    'RandomStream','mt19937ar with seed', ...
                    'Seed', file_seed);
                reset(chan);
                x_fad = chan(x);
            case 'rician'
                chan = comm.RicianChannel( ...
                    'SampleRate', Fs, ...
                    'KFactor', K_linear, ...     % ★ 线性 K
                    'PathDelays', 0, ...
                    'AveragePathGains', 0, ...
                    'MaximumDopplerShift', fd_hz, ...
                    'RandomStream','mt19937ar with seed', ...
                    'Seed', file_seed);
                reset(chan);
                x_fad = chan(x);
            otherwise
                error('未知 fading_type: %s', fading_type);
        end

        % ---------- 信道失真能量（相对输入 x） ----------
        e_chan = x_fad - x;
        E_chan = sum(abs(e_chan).^2);                      % 信道已贡献的“误差能量”
        chan_err_frac = E_chan / (E_chan + eps);           % 仅为输出方便（下面也会写 CSV）

        % ---------- 针对每个“衰减 ΔdB”循环 ----------
        for deg_tgt = snr_deg_list
            snr_dir = fullfile(out_dir, sprintf('SNRdeg_%ddB', deg_tgt));
            if ~exist(snr_dir, 'dir'); mkdir(snr_dir); end

            % 目标总“误差能量” E_err_target = P_sig / 10^(Δ/10)
            E_err_target = P_sig / (10^(deg_tgt/10));

            % 由噪声补齐的能量：E_noise = max(E_err_target - E_chan, 0)
            E_noise = max(E_err_target - E_chan, 0);
            if E_noise == 0
                % 信道自身造成的失真已>=目标衰减所需误差能量；不再加噪
                noise = zeros(N,1);
                noise_pow_per_samp = 0;
            else
                % 复高斯噪声：Var = E_noise / N；实虚各一半
                var_cplx = E_noise / N;
                sigma    = sqrt(var_cplx/2);
                rng(file_seed + round(1e6*deg_tgt));       % 每档 ΔdB 不同子种子
                noise = sigma*(randn(N,1) + 1j*randn(N,1));
                noise_pow_per_samp = var_cplx;
            end

            % 输出信号
            y = x_fad + noise;

            % —— 验证“相对输入的 SNR 衰减”是否达标 —— 
            E_err_eff = sum(abs(y - x).^2);
            deg_eff   = 10*log10( P_sig / max(E_err_eff, eps) );

            fprintf('   Δ_target=%4.0f dB,  Δ_eff=%7.3f dB  (E_chan/target=%6.2f%%)\n', ...
                    deg_tgt, deg_eff, 100*E_chan/max(E_err_target,eps));
            fprintf(fid, '%s,%g,%.6f,%.6f,%.6e\n', stem, deg_tgt, deg_eff, E_chan/max(E_err_target,eps), noise_pow_per_samp);

            % ---------- 时频图 ----------
            [Sxx, F, T] = spectrogram(y, win_func, overlap, nfft, Fs, 'centered');
            P_dB = 20*log10(abs(Sxx) + eps);

            % 分位数归一化（跨样本更稳，可按需改回 min-max）
            lo = prctile(P_dB(:), 1); hi = prctile(P_dB(:), 99);
            P01 = (P_dB - lo) / max(hi - lo, eps);
            P01 = min(max(P01,0),1);

            % 尺寸统一
            P_rs = imresize(P01, [desired_rows, desired_cols], 'bicubic');
            P_out = P_rs.';   % 若你下游约定是转置后的 [W×H]，保留此行

            % 保存 PNG（jet 伪彩）
            cmap = jet(256);
            idxImg = min(max(round(P_out*255)+1,1),256);
            RGB = ind2rgb(uint8(idxImg), cmap);
            out_png = fullfile(snr_dir, sprintf('%s_Deg%d.png', stem, deg_tgt));
            imwrite(RGB, out_png);
        end

        fprintf('文件 "%s" 完成。\n\n', mat_file.name);
    end

    fclose(fid);
    fprintf('子文件夹 "%s" 处理结束。日志：%s\n\n', sub_name, log_csv);
end

fprintf('全部完成。\n');
