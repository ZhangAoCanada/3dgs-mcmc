for scale_reg in 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
do
    noise_lr=5e3
    cap_max=10000000
    densify_from_iter=500
    densification_interval=400
    echo "[TRAINING] training with cap_max=${cap_max} and noise_lr=${noise_lr}"
    CUDA_VISIBLE_DEVICES=3 python train.py --source_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train --test_path data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test --model_path outputs/mcmc/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/original_sfm_${cap_max}_${noise_lr}_scale${scale_reg}_opacity0.01_densification-${densify_from_iter}-${densification_interval} --images input_cached --resolution -1 --init_type sfm --cap_max ${cap_max} --data_device cpu --scale_reg ${scale_reg} --opacity_reg 0.01 --noise_lr ${noise_lr} --densify_from_iter ${densify_from_iter} --densification_interval ${densification_interval}
done