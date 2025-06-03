
# 最終修正版 DAC統合コード
def codes_to_audio(self, audio_codes):
    '''音声コードから実際の音声波形を生成 (最終版)'''
    
    print(f"🎵 Converting codes to audio...")
    print(f"   Input codes shape: {audio_codes.shape}")
    
    with torch.no_grad():
        start_time = time.time()
        
        try:
            # DAC model の内部モデルを取得（重要！）
            dac_internal = self.audio_encoder.model
            quantizer = dac_internal.quantizer
            
            # 音声コードの形状を確認・調整
            if audio_codes.dim() == 2:
                audio_codes = audio_codes.unsqueeze(0)  # [9, 100] -> [1, 9, 100]
            
            # Quantizerで音声コードを潜在表現に変換
            # from_codes または embed メソッドを使用
            if hasattr(quantizer, 'from_codes'):
                latents = quantizer.from_codes(audio_codes)
            elif hasattr(quantizer, 'embed'):
                latents = quantizer.embed(audio_codes)
            else:
                raise Exception("No suitable quantizer method found")
            
            print(f"   Latents shape: {latents.shape}")
            
            # DAC decode実行
            audio_waveform = dac_internal.decode(latents)
            print(f"   Audio waveform shape: {audio_waveform.shape}")
            
            if isinstance(audio_waveform, torch.Tensor):
                audio_np = audio_waveform.squeeze().cpu().numpy()
            else:
                audio_np = np.array(audio_waveform)
            
            end_time = time.time()
            
            print(f"   ⏱️  Audio conversion time: {end_time - start_time:.3f}s")
            print(f"   🎚️  Audio range: [{audio_np.min():.3f}, {audio_np.max():.3f}]")
            
            return {
                'audio_waveform': audio_np,
                'conversion_time': end_time - start_time,
                'sample_rate': self.dac_config.sampling_rate,
                'duration': len(audio_np) / self.dac_config.sampling_rate
            }
            
        except Exception as e:
            print(f"   ❌ DAC decoding failed: {e}")
            print(f"   🔄 Falling back to mock audio generation...")
            return self.fallback_mock_audio()
