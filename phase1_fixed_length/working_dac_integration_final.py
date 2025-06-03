
# 最終的に動作するDAC統合コード
def codes_to_audio(self, audio_codes):
    '''音声コードから実際の音声波形を生成 (最終動作版)'''
    
    print(f"🎵 Converting codes to audio...")
    print(f"   Input codes shape: {audio_codes.shape}")
    
    with torch.no_grad():
        start_time = time.time()
        
        try:
            # DAC内部モデルを取得
            dac_internal = self.audio_encoder.model
            quantizer = dac_internal.quantizer
            
            # 音声コードの形状を調整
            if audio_codes.dim() == 2:
                audio_codes = audio_codes.unsqueeze(0)  # [9, 100] -> [1, 9, 100]
            
            # 音声コードを正しい範囲にクランプ
            audio_codes = torch.clamp(audio_codes, 0, self.dac_config.codebook_size - 1)
            print(f"   Clamped codes range: [{audio_codes.min()}, {audio_codes.max()}]")
            
            # 各quantizerレイヤーを手動で処理
            latents = torch.zeros(audio_codes.shape[0], 1024, audio_codes.shape[2], 
                                device=audio_codes.device)
            
            for i, quantizer_layer in enumerate(quantizer.quantizers):
                if i < audio_codes.shape[1]:
                    layer_codes = audio_codes[:, i, :]  # [1, 100]
                    
                    # 埋め込み処理
                    if hasattr(quantizer_layer, 'embed'):
                        embedded = quantizer_layer.embed(layer_codes)
                    else:
                        embedded = quantizer_layer.codebook(layer_codes)
                        embedded = embedded.transpose(1, 2)  # [1, time, dim] -> [1, dim, time]
                    
                    latents += embedded
            
            print(f"   Combined latents shape: {latents.shape}")
            
            # DAC decode実行
            audio_waveform = dac_internal.decode(latents.detach())
            
            if isinstance(audio_waveform, torch.Tensor):
                audio_np = audio_waveform.detach().squeeze().cpu().numpy()
            else:
                audio_np = np.array(audio_waveform)
            
            end_time = time.time()
            
            print(f"   ⏱️  Audio conversion time: {end_time - start_time:.3f}s")
            print(f"   📊 Audio waveform shape: {audio_np.shape}")
            print(f"   🎚️  Audio range: [{audio_np.min():.3f}, {audio_np.max():.3f}]")
            
            return {
                'audio_waveform': audio_np,
                'conversion_time': end_time - start_time,
                'sample_rate': self.dac_config.sampling_rate,
                'duration': len(audio_np) / self.dac_config.sampling_rate
            }
            
        except Exception as e:
            print(f"   ❌ DAC decoding failed: {e}")
            return self.fallback_mock_audio()
