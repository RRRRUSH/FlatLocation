```bash
python E:\HaozhanLi\Project\FlatLoc\EqNIO\TLIO-master\src\main_filter.py `
--model_path E:\HaozhanLi\Project\FlatLoc\EqNIO\models\TLIO_O2_openbody\checkpoints\checkpoint_43.pt `
--model_param_path E:\HaozhanLi\Project\FlatLoc\EqNIO\models\TLIO_O2_openbody\parameters.json `
--root_dir E:\HaozhanLi\Project\FlatLoc\EqNIO\data\flact3_rect `
--out_dir E:\HaozhanLi\Project\FlatLoc\EqNIO\output\0409\epoch43_openAndUs_wq_initbabg_flact3_rect `
--save_as_npy `
--sigma_na 1.1212224622294688e-02 `
--sigma_ng 1.1752829958889025e-03 `
--ita_ba 3.4661445096824021e-04 `
--ita_bg 5.6034859604820343e-06 `
--init_ba_sigma 0.001 `
--init_bg_sigma 0.00001
```

python E:\HaozhanLi\Project\FlatLoc\EqNIO\TLIO-master\src\main_filter.py --root_dir E:\HaozhanLi\Project\FlatLoc\EqNIO\data\tx\test_radar_body --out_dir E:\HaozhanLi\Project\FlatLoc\EqNIO\output\0410\test_radar_body --model_path E:\HaozhanLi\Project\FlatLoc\EqNIO\models\TLIO_O2_openbody\checkpoints\checkpoint_43.pt --model_param_path E:\HaozhanLi\Project\FlatLoc\EqNIO\models\TLIO_O2_openbody\parameters.json --save_as_npy 