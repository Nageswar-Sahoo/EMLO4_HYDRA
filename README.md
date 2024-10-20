<h1>Dog Breed Image Dataset Training, Evaluation, and Inference with Docker </h1>

This repository contains a PyTorch Lightning-based project for classifying dog breeds using a dataset from Kaggle. The project includes Docker support, a DevContainer setup, and inference using a pre-trained model. Configuration management is handled using Hydra.

Features
   
           1> Train and evaluate a deep learning model on the Dog Breed Image Dataset
	   
           2> Docker containerization for easy deployment and reproducibility
	   	   
           3> Hydra for flexible configuration management
	   
           4> eval.py to run evaluation on the validation dataset and print metrics
	   
           5> infer.py for inference on sample images

<h3>About Dataset</h3>

Description

This dataset contains a collection of images for 10 different dog breeds, meticulously gathered and organized to facilitate various computer vision tasks such as image classification and object detection. The dataset includes the following breeds:

       Golden Retriever
       German Shepherd
       Labrador Retriever
       Bulldog
       Beagle
       Poodle
       Rottweiler
       Yorkshire Terrier
       Boxer
       Dachshund


<h2>Using Hydra for Configuration Management</h2>
The project utilizes Hydra to manage configurations. Configuration files are located in the configs/ directory. You can modify these files to adjust various parameters for training, evaluation, and inference.

Running with Hydra
To run the training script with a specified configuration, use the following command:

        python src/train.py 
	    python src/eval.py 
        python src/infer.py 

All Hydra configurations are located in the following directory. Any updates to the parameters can be made there.

<img width="886" alt="image" src="https://github.com/user-attachments/assets/4150a98c-7f47-44b5-a388-fc11f2ac831a">

 



Model  Summery : 
             ┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
             ┃     ┃ Name                           ┃ Type                      ┃ Params ┃ Mode  ┃
             ┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
             │ 0   │ model                          │ VisionTransformer         │  5.5 M │ train │
│ 1   │ model.patch_embed              │ PatchEmbed                │  147 K │ train │
│ 2   │ model.patch_embed.proj         │ Conv2d                    │  147 K │ train │
│ 3   │ model.patch_embed.norm         │ Identity                  │      0 │ train │
│ 4   │ model.pos_drop                 │ Dropout                   │      0 │ train │
│ 5   │ model.patch_drop               │ Identity                  │      0 │ train │
│ 6   │ model.norm_pre                 │ Identity                  │      0 │ train │
│ 7   │ model.blocks                   │ Sequential                │  5.3 M │ train │
│ 8   │ model.blocks.0                 │ Block                     │  444 K │ train │
│ 9   │ model.blocks.0.norm1           │ LayerNorm                 │    384 │ train │
│ 10  │ model.blocks.0.attn            │ Attention                 │  148 K │ train │
│ 11  │ model.blocks.0.attn.qkv        │ Linear                    │  111 K │ train │
│ 12  │ model.blocks.0.attn.q_norm     │ Identity                  │      0 │ train │
│ 13  │ model.blocks.0.attn.k_norm     │ Identity                  │      0 │ train │
│ 14  │ model.blocks.0.attn.attn_drop  │ Dropout                   │      0 │ train │
│ 15  │ model.blocks.0.attn.proj       │ Linear                    │ 37.1 K │ train │
│ 16  │ model.blocks.0.attn.proj_drop  │ Dropout                   │      0 │ train │
│ 17  │ model.blocks.0.ls1             │ Identity                  │      0 │ train │
│ 18  │ model.blocks.0.drop_path1      │ Identity                  │      0 │ train │
│ 19  │ model.blocks.0.norm2           │ LayerNorm                 │    384 │ train │
│ 20  │ model.blocks.0.mlp             │ Mlp                       │  295 K │ train │
│ 21  │ model.blocks.0.mlp.fc1         │ Linear                    │  148 K │ train │
│ 22  │ model.blocks.0.mlp.act         │ GELU                      │      0 │ train │
│ 23  │ model.blocks.0.mlp.drop1       │ Dropout                   │      0 │ train │
│ 24  │ model.blocks.0.mlp.norm        │ Identity                  │      0 │ train │
│ 25  │ model.blocks.0.mlp.fc2         │ Linear                    │  147 K │ train │
│ 26  │ model.blocks.0.mlp.drop2       │ Dropout                   │      0 │ train │
│ 27  │ model.blocks.0.ls2             │ Identity                  │      0 │ train │
│ 28  │ model.blocks.0.drop_path2      │ Identity                  │      0 │ train │
│ 29  │ model.blocks.1                 │ Block                     │  444 K │ train │
│ 30  │ model.blocks.1.norm1           │ LayerNorm                 │    384 │ train │
│ 31  │ model.blocks.1.attn            │ Attention                 │  148 K │ train │
│ 32  │ model.blocks.1.attn.qkv        │ Linear                    │  111 K │ train │
│ 33  │ model.blocks.1.attn.q_norm     │ Identity                  │      0 │ train │
│ 34  │ model.blocks.1.attn.k_norm     │ Identity                  │      0 │ train │
│ 35  │ model.blocks.1.attn.attn_drop  │ Dropout                   │      0 │ train │
│ 36  │ model.blocks.1.attn.proj       │ Linear                    │ 37.1 K │ train │
│ 37  │ model.blocks.1.attn.proj_drop  │ Dropout                   │      0 │ train │
│ 38  │ model.blocks.1.ls1             │ Identity                  │      0 │ train │
│ 39  │ model.blocks.1.drop_path1      │ Identity                  │      0 │ train │
│ 40  │ model.blocks.1.norm2           │ LayerNorm                 │    384 │ train │
│ 41  │ model.blocks.1.mlp             │ Mlp                       │  295 K │ train │
│ 42  │ model.blocks.1.mlp.fc1         │ Linear                    │  148 K │ train │
│ 43  │ model.blocks.1.mlp.act         │ GELU                      │      0 │ train │
│ 44  │ model.blocks.1.mlp.drop1       │ Dropout                   │      0 │ train │
│ 45  │ model.blocks.1.mlp.norm        │ Identity                  │      0 │ train │
│ 46  │ model.blocks.1.mlp.fc2         │ Linear                    │  147 K │ train │
│ 47  │ model.blocks.1.mlp.drop2       │ Dropout                   │      0 │ train │
│ 48  │ model.blocks.1.ls2             │ Identity                  │      0 │ train │
│ 49  │ model.blocks.1.drop_path2      │ Identity                  │      0 │ train │
│ 50  │ model.blocks.2                 │ Block                     │  444 K │ train │
│ 51  │ model.blocks.2.norm1           │ LayerNorm                 │    384 │ train │
│ 52  │ model.blocks.2.attn            │ Attention                 │  148 K │ train │
│ 53  │ model.blocks.2.attn.qkv        │ Linear                    │  111 K │ train │
│ 54  │ model.blocks.2.attn.q_norm     │ Identity                  │      0 │ train │
│ 55  │ model.blocks.2.attn.k_norm     │ Identity                  │      0 │ train │
│ 56  │ model.blocks.2.attn.attn_drop  │ Dropout                   │      0 │ train │
│ 57  │ model.blocks.2.attn.proj       │ Linear                    │ 37.1 K │ train │
│ 58  │ model.blocks.2.attn.proj_drop  │ Dropout                   │      0 │ train │
│ 59  │ model.blocks.2.ls1             │ Identity                  │      0 │ train │
│ 60  │ model.blocks.2.drop_path1      │ Identity                  │      0 │ train │
│ 61  │ model.blocks.2.norm2           │ LayerNorm                 │    384 │ train │
│ 62  │ model.blocks.2.mlp             │ Mlp                       │  295 K │ train │
│ 63  │ model.blocks.2.mlp.fc1         │ Linear                    │  148 K │ train │
│ 64  │ model.blocks.2.mlp.act         │ GELU                      │      0 │ train │
│ 65  │ model.blocks.2.mlp.drop1       │ Dropout                   │      0 │ train │
│ 66  │ model.blocks.2.mlp.norm        │ Identity                  │      0 │ train │
│ 67  │ model.blocks.2.mlp.fc2         │ Linear                    │  147 K │ train │
│ 68  │ model.blocks.2.mlp.drop2       │ Dropout                   │      0 │ train │
│ 69  │ model.blocks.2.ls2             │ Identity                  │      0 │ train │
│ 70  │ model.blocks.2.drop_path2      │ Identity                  │      0 │ train │
│ 71  │ model.blocks.3                 │ Block                     │  444 K │ train │
│ 72  │ model.blocks.3.norm1           │ LayerNorm                 │    384 │ train │
│ 73  │ model.blocks.3.attn            │ Attention                 │  148 K │ train │
│ 74  │ model.blocks.3.attn.qkv        │ Linear                    │  111 K │ train │
│ 75  │ model.blocks.3.attn.q_norm     │ Identity                  │      0 │ train │
│ 76  │ model.blocks.3.attn.k_norm     │ Identity                  │      0 │ train │
│ 77  │ model.blocks.3.attn.attn_drop  │ Dropout                   │      0 │ train │
│ 78  │ model.blocks.3.attn.proj       │ Linear                    │ 37.1 K │ train │
│ 79  │ model.blocks.3.attn.proj_drop  │ Dropout                   │      0 │ train │
│ 80  │ model.blocks.3.ls1             │ Identity                  │      0 │ train │
│ 81  │ model.blocks.3.drop_path1      │ Identity                  │      0 │ train │
│ 82  │ model.blocks.3.norm2           │ LayerNorm                 │    384 │ train │
│ 83  │ model.blocks.3.mlp             │ Mlp                       │  295 K │ train │
│ 84  │ model.blocks.3.mlp.fc1         │ Linear                    │  148 K │ train │
│ 85  │ model.blocks.3.mlp.act         │ GELU                      │      0 │ train │
│ 86  │ model.blocks.3.mlp.drop1       │ Dropout                   │      0 │ train │
│ 87  │ model.blocks.3.mlp.norm        │ Identity                  │      0 │ train │
│ 88  │ model.blocks.3.mlp.fc2         │ Linear                    │  147 K │ train │
│ 89  │ model.blocks.3.mlp.drop2       │ Dropout                   │      0 │ train │
│ 90  │ model.blocks.3.ls2             │ Identity                  │      0 │ train │
│ 91  │ model.blocks.3.drop_path2      │ Identity                  │      0 │ train │
│ 92  │ model.blocks.4                 │ Block                     │  444 K │ train │
│ 93  │ model.blocks.4.norm1           │ LayerNorm                 │    384 │ train │
│ 94  │ model.blocks.4.attn            │ Attention                 │  148 K │ train │
│ 95  │ model.blocks.4.attn.qkv        │ Linear                    │  111 K │ train │
│ 96  │ model.blocks.4.attn.q_norm     │ Identity                  │      0 │ train │
│ 97  │ model.blocks.4.attn.k_norm     │ Identity                  │      0 │ train │
│ 98  │ model.blocks.4.attn.attn_drop  │ Dropout                   │      0 │ train │
│ 99  │ model.blocks.4.attn.proj       │ Linear                    │ 37.1 K │ train │
│ 100 │ model.blocks.4.attn.proj_drop  │ Dropout                   │      0 │ train │
│ 101 │ model.blocks.4.ls1             │ Identity                  │      0 │ train │
│ 102 │ model.blocks.4.drop_path1      │ Identity                  │      0 │ train │
│ 103 │ model.blocks.4.norm2           │ LayerNorm                 │    384 │ train │
│ 104 │ model.blocks.4.mlp             │ Mlp                       │  295 K │ train │
│ 105 │ model.blocks.4.mlp.fc1         │ Linear                    │  148 K │ train │
│ 106 │ model.blocks.4.mlp.act         │ GELU                      │      0 │ train │
│ 107 │ model.blocks.4.mlp.drop1       │ Dropout                   │      0 │ train │
│ 108 │ model.blocks.4.mlp.norm        │ Identity                  │      0 │ train │
│ 109 │ model.blocks.4.mlp.fc2         │ Linear                    │  147 K │ train │
│ 110 │ model.blocks.4.mlp.drop2       │ Dropout                   │      0 │ train │
│ 111 │ model.blocks.4.ls2             │ Identity                  │      0 │ train │
│ 112 │ model.blocks.4.drop_path2      │ Identity                  │      0 │ train │
│ 113 │ model.blocks.5                 │ Block                     │  444 K │ train │
│ 114 │ model.blocks.5.norm1           │ LayerNorm                 │    384 │ train │
│ 115 │ model.blocks.5.attn            │ Attention                 │  148 K │ train │
│ 116 │ model.blocks.5.attn.qkv        │ Linear                    │  111 K │ train │
│ 117 │ model.blocks.5.attn.q_norm     │ Identity                  │      0 │ train │
│ 118 │ model.blocks.5.attn.k_norm     │ Identity                  │      0 │ train │
│ 119 │ model.blocks.5.attn.attn_drop  │ Dropout                   │      0 │ train │
│ 120 │ model.blocks.5.attn.proj       │ Linear                    │ 37.1 K │ train │
│ 121 │ model.blocks.5.attn.proj_drop  │ Dropout                   │      0 │ train │
│ 122 │ model.blocks.5.ls1             │ Identity                  │      0 │ train │
│ 123 │ model.blocks.5.drop_path1      │ Identity                  │      0 │ train │
│ 124 │ model.blocks.5.norm2           │ LayerNorm                 │    384 │ train │
│ 125 │ model.blocks.5.mlp             │ Mlp                       │  295 K │ train │
│ 126 │ model.blocks.5.mlp.fc1         │ Linear                    │  148 K │ train │
│ 127 │ model.blocks.5.mlp.act         │ GELU                      │      0 │ train │
│ 128 │ model.blocks.5.mlp.drop1       │ Dropout                   │      0 │ train │
│ 129 │ model.blocks.5.mlp.norm        │ Identity                  │      0 │ train │
│ 130 │ model.blocks.5.mlp.fc2         │ Linear                    │  147 K │ train │
│ 131 │ model.blocks.5.mlp.drop2       │ Dropout                   │      0 │ train │
│ 132 │ model.blocks.5.ls2             │ Identity                  │      0 │ train │
│ 133 │ model.blocks.5.drop_path2      │ Identity                  │      0 │ train │
│ 134 │ model.blocks.6                 │ Block                     │  444 K │ train │
│ 135 │ model.blocks.6.norm1           │ LayerNorm                 │    384 │ train │
│ 136 │ model.blocks.6.attn            │ Attention                 │  148 K │ train │
│ 137 │ model.blocks.6.attn.qkv        │ Linear                    │  111 K │ train │
│ 138 │ model.blocks.6.attn.q_norm     │ Identity                  │      0 │ train │
│ 139 │ model.blocks.6.attn.k_norm     │ Identity                  │      0 │ train │
│ 140 │ model.blocks.6.attn.attn_drop  │ Dropout                   │      0 │ train │
│ 141 │ model.blocks.6.attn.proj       │ Linear                    │ 37.1 K │ train │
│ 142 │ model.blocks.6.attn.proj_drop  │ Dropout                   │      0 │ train │
│ 143 │ model.blocks.6.ls1             │ Identity                  │      0 │ train │
│ 144 │ model.blocks.6.drop_path1      │ Identity                  │      0 │ train │
│ 145 │ model.blocks.6.norm2           │ LayerNorm                 │    384 │ train │
│ 146 │ model.blocks.6.mlp             │ Mlp                       │  295 K │ train │
│ 147 │ model.blocks.6.mlp.fc1         │ Linear                    │  148 K │ train │
│ 148 │ model.blocks.6.mlp.act         │ GELU                      │      0 │ train │
│ 149 │ model.blocks.6.mlp.drop1       │ Dropout                   │      0 │ train │
│ 150 │ model.blocks.6.mlp.norm        │ Identity                  │      0 │ train │
│ 151 │ model.blocks.6.mlp.fc2         │ Linear                    │  147 K │ train │
│ 152 │ model.blocks.6.mlp.drop2       │ Dropout                   │      0 │ train │
│ 153 │ model.blocks.6.ls2             │ Identity                  │      0 │ train │
             │ 154 │ model.blocks.6.drop_path2      │ Identity                  │      0 │ train │
             │ 155 │ model.blocks.7                 │ Block                     │  444 K │ train │
             │ 156 │ model.blocks.7.norm1           │ LayerNorm                 │    384 │ train │
             │ 157 │ model.blocks.7.attn            │ Attention                 │  148 K │ train │
             │ 158 │ model.blocks.7.attn.qkv        │ Linear                    │  111 K │ train │
             │ 159 │ model.blocks.7.attn.q_norm     │ Identity                  │      0 │ train │
             │ 160 │ model.blocks.7.attn.k_norm     │ Identity                  │      0 │ train │
             │ 161 │ model.blocks.7.attn.attn_drop  │ Dropout                   │      0 │ train │
             │ 162 │ model.blocks.7.attn.proj       │ Linear                    │ 37.1 K │ train │
             │ 163 │ model.blocks.7.attn.proj_drop  │ Dropout                   │      0 │ train │
             │ 164 │ model.blocks.7.ls1             │ Identity                  │      0 │ train │
             │ 165 │ model.blocks.7.drop_path1      │ Identity                  │      0 │ train │
             │ 166 │ model.blocks.7.norm2           │ LayerNorm                 │    384 │ train │
             │ 167 │ model.blocks.7.mlp             │ Mlp                       │  295 K │ train │
             │ 168 │ model.blocks.7.mlp.fc1         │ Linear                    │  148 K │ train │
             │ 169 │ model.blocks.7.mlp.act         │ GELU                      │      0 │ train │
             │ 170 │ model.blocks.7.mlp.drop1       │ Dropout                   │      0 │ train │
             │ 171 │ model.blocks.7.mlp.norm        │ Identity                  │      0 │ train │
             │ 172 │ model.blocks.7.mlp.fc2         │ Linear                    │  147 K │ train │
             │ 173 │ model.blocks.7.mlp.drop2       │ Dropout                   │      0 │ train │
             │ 174 │ model.blocks.7.ls2             │ Identity                  │      0 │ train │
             │ 175 │ model.blocks.7.drop_path2      │ Identity                  │      0 │ train │
             │ 176 │ model.blocks.8                 │ Block                     │  444 K │ train │
             │ 177 │ model.blocks.8.norm1           │ LayerNorm                 │    384 │ train │
             │ 178 │ model.blocks.8.attn            │ Attention                 │  148 K │ train │
             │ 179 │ model.blocks.8.attn.qkv        │ Linear                    │  111 K │ train │
             │ 180 │ model.blocks.8.attn.q_norm     │ Identity                  │      0 │ train │
             │ 181 │ model.blocks.8.attn.k_norm     │ Identity                  │      0 │ train │
             │ 182 │ model.blocks.8.attn.attn_drop  │ Dropout                   │      0 │ train │
             │ 183 │ model.blocks.8.attn.proj       │ Linear                    │ 37.1 K │ train │
             │ 184 │ model.blocks.8.attn.proj_drop  │ Dropout                   │      0 │ train │
             │ 185 │ model.blocks.8.ls1             │ Identity                  │      0 │ train │
             │ 186 │ model.blocks.8.drop_path1      │ Identity                  │      0 │ train │
             │ 187 │ model.blocks.8.norm2           │ LayerNorm                 │    384 │ train │
             │ 188 │ model.blocks.8.mlp             │ Mlp                       │  295 K │ train │
             │ 189 │ model.blocks.8.mlp.fc1         │ Linear                    │  148 K │ train │
             │ 190 │ model.blocks.8.mlp.act         │ GELU                      │      0 │ train │
             │ 191 │ model.blocks.8.mlp.drop1       │ Dropout                   │      0 │ train │
             │ 192 │ model.blocks.8.mlp.norm        │ Identity                  │      0 │ train │
             │ 193 │ model.blocks.8.mlp.fc2         │ Linear                    │  147 K │ train │
             │ 194 │ model.blocks.8.mlp.drop2       │ Dropout                   │      0 │ train │
             │ 195 │ model.blocks.8.ls2             │ Identity                  │      0 │ train │
             │ 196 │ model.blocks.8.drop_path2      │ Identity                  │      0 │ train │
             │ 197 │ model.blocks.9                 │ Block                     │  444 K │ train │
             │ 198 │ model.blocks.9.norm1           │ LayerNorm                 │    384 │ train │
             │ 199 │ model.blocks.9.attn            │ Attention                 │  148 K │ train │
             │ 200 │ model.blocks.9.attn.qkv        │ Linear                    │  111 K │ train │
             │ 201 │ model.blocks.9.attn.q_norm     │ Identity                  │      0 │ train │
             │ 202 │ model.blocks.9.attn.k_norm     │ Identity                  │      0 │ train │
             │ 203 │ model.blocks.9.attn.attn_drop  │ Dropout                   │      0 │ train │
             │ 204 │ model.blocks.9.attn.proj       │ Linear                    │ 37.1 K │ train │
             │ 205 │ model.blocks.9.attn.proj_drop  │ Dropout                   │      0 │ train │
             │ 206 │ model.blocks.9.ls1             │ Identity                  │      0 │ train │
             │ 207 │ model.blocks.9.drop_path1      │ Identity                  │      0 │ train │
             │ 208 │ model.blocks.9.norm2           │ LayerNorm                 │    384 │ train │
             │ 209 │ model.blocks.9.mlp             │ Mlp                       │  295 K │ train │
             │ 210 │ model.blocks.9.mlp.fc1         │ Linear                    │  148 K │ train │
             │ 211 │ model.blocks.9.mlp.act         │ GELU                      │      0 │ train │
             │ 212 │ model.blocks.9.mlp.drop1       │ Dropout                   │      0 │ train │
             │ 213 │ model.blocks.9.mlp.norm        │ Identity                  │      0 │ train │
             │ 214 │ model.blocks.9.mlp.fc2         │ Linear                    │  147 K │ train │
             │ 215 │ model.blocks.9.mlp.drop2       │ Dropout                   │      0 │ train │
             │ 216 │ model.blocks.9.ls2             │ Identity                  │      0 │ train │
             │ 217 │ model.blocks.9.drop_path2      │ Identity                  │      0 │ train │
             │ 218 │ model.blocks.10                │ Block                     │  444 K │ train │
             │ 219 │ model.blocks.10.norm1          │ LayerNorm                 │    384 │ train │
             │ 220 │ model.blocks.10.attn           │ Attention                 │  148 K │ train │
             │ 221 │ model.blocks.10.attn.qkv       │ Linear                    │  111 K │ train │
             │ 222 │ model.blocks.10.attn.q_norm    │ Identity                  │      0 │ train │
             │ 223 │ model.blocks.10.attn.k_norm    │ Identity                  │      0 │ train │
             │ 224 │ model.blocks.10.attn.attn_drop │ Dropout                   │      0 │ train │
             │ 225 │ model.blocks.10.attn.proj      │ Linear                    │ 37.1 K │ train │
             │ 226 │ model.blocks.10.attn.proj_drop │ Dropout                   │      0 │ train │
             │ 227 │ model.blocks.10.ls1            │ Identity                  │      0 │ train │
             │ 228 │ model.blocks.10.drop_path1     │ Identity                  │      0 │ train │
             │ 229 │ model.blocks.10.norm2          │ LayerNorm                 │    384 │ train │
             │ 230 │ model.blocks.10.mlp            │ Mlp                       │  295 K │ train │
             │ 231 │ model.blocks.10.mlp.fc1        │ Linear                    │  148 K │ train │
             │ 232 │ model.blocks.10.mlp.act        │ GELU                      │      0 │ train │
             │ 233 │ model.blocks.10.mlp.drop1      │ Dropout                   │      0 │ train │
             │ 234 │ model.blocks.10.mlp.norm       │ Identity                  │      0 │ train │
             │ 235 │ model.blocks.10.mlp.fc2        │ Linear                    │  147 K │ train │
             │ 236 │ model.blocks.10.mlp.drop2      │ Dropout                   │      0 │ train │
             │ 237 │ model.blocks.10.ls2            │ Identity                  │      0 │ train │
             │ 238 │ model.blocks.10.drop_path2     │ Identity                  │      0 │ train │
             │ 239 │ model.blocks.11                │ Block                     │  444 K │ train │
             │ 240 │ model.blocks.11.norm1          │ LayerNorm                 │    384 │ train │
             │ 241 │ model.blocks.11.attn           │ Attention                 │  148 K │ train │
             │ 242 │ model.blocks.11.attn.qkv       │ Linear                    │  111 K │ train │
             │ 243 │ model.blocks.11.attn.q_norm    │ Identity                  │      0 │ train │
             │ 244 │ model.blocks.11.attn.k_norm    │ Identity                  │      0 │ train │
             │ 245 │ model.blocks.11.attn.attn_drop │ Dropout                   │      0 │ train │
             │ 246 │ model.blocks.11.attn.proj      │ Linear                    │ 37.1 K │ train │
             │ 247 │ model.blocks.11.attn.proj_drop │ Dropout                   │      0 │ train │
             │ 248 │ model.blocks.11.ls1            │ Identity                  │      0 │ train │
             │ 249 │ model.blocks.11.drop_path1     │ Identity                  │      0 │ train │
             │ 250 │ model.blocks.11.norm2          │ LayerNorm                 │    384 │ train │
             │ 251 │ model.blocks.11.mlp            │ Mlp                       │  295 K │ train │
             │ 252 │ model.blocks.11.mlp.fc1        │ Linear                    │  148 K │ train │
             │ 253 │ model.blocks.11.mlp.act        │ GELU                      │      0 │ train │
             │ 254 │ model.blocks.11.mlp.drop1      │ Dropout                   │      0 │ train │
             │ 255 │ model.blocks.11.mlp.norm       │ Identity                  │      0 │ train │
             │ 256 │ model.blocks.11.mlp.fc2        │ Linear                    │  147 K │ train │
             │ 257 │ model.blocks.11.mlp.drop2      │ Dropout                   │      0 │ train │
             │ 258 │ model.blocks.11.ls2            │ Identity                  │      0 │ train │
             │ 259 │ model.blocks.11.drop_path2     │ Identity                  │      0 │ train │
             │ 260 │ model.norm                     │ LayerNorm                 │    384 │ train │
             │ 261 │ model.fc_norm                  │ Identity                  │      0 │ train │
             │ 262 │ model.head_drop                │ Dropout                   │      0 │ train │
             │ 263 │ model.head                     │ Linear                    │    386 │ train │
             │ 264 │ train_acc                      │ MulticlassAccuracy        │      0 │ train │
             │ 265 │ val_acc                        │ MulticlassAccuracy        │      0 │ train │
             │ 266 │ test_acc                       │ MulticlassAccuracy        │      0 │ train │
             │ 267 │ train_conf_matrix              │ MulticlassConfusionMatrix │      0 │ train │
             │ 268 │ val_conf_matrix                │ MulticlassConfusionMatrix │      0 │ train │
             └─────┴────────────────────────────────┴───────────────────────────┴────────┴───────┘

How to Train, Evaluate, and Infer Using Docker

 1. Build the Docker image:

          docker build -t dogbreed-classification .
 
 2. To run training:

          docker run -v $(pwd)/model_artifacts:/app/logs dogbreed-classification src/train.py

          Above docker script will generate the model artifact in below directory .
    
 <img width="470" alt="image" src="https://github.com/user-attachments/assets/6f518288-0ecd-4995-8eb9-4a73319d6f0f">
 
 3. To run evaluation:

          docker run -v $(pwd)/model_artifacts:/app/logs dogbreed-classification src/eval.py ckpt_path=./logs/train/runs/2024-10-08_10-50-43/checkpoints/epoch_001.ckpt

          please update the best model check point path generated from training script . 
    
 4. To run inference:

          docker run -v $(pwd)/model_artifacts:/app/logs dogbreed-classification src/infer.py ckpt_path=./logs/train/runs/2024-10-08_10-50-43/checkpoints/epoch_001.ckpt

          please update the best model check point path generated from training script . 


 5 By default it performs inference on the images present in the dataset folder.

          To modify the infer arguments, you can do the following:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification infer --input_folder="path/to/custom/input" --  output_folder="path/to/custom/output" --     ckpt_path="path/to/custom/checkpoint.ckpt"



<h3>Docker Setup</h3>

1. Dockerfile
This repository includes a Dockerfile to containerize the training, evaluation, and inference process. The Docker image includes the necessary dependencies and installs the project package.
         # Build stage
         FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

         ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
         ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

         WORKDIR /app

         # Install dependencies
          RUN --mount=type=cache,target=/root/.cache/uv \
         	--mount=type=bind,source=uv.lock,target=uv.lock \
        	--mount=type=bind,source=pyproject.toml,target=pyproject.toml \
        	uv sync --frozen --no-install-project --no-dev

         # Copy the rest of the application
         ADD . /app

         # Install the project and its dependencies
         RUN --mount=type=cache,target=/root/.cache/uv \
     	uv sync --frozen --no-dev

         # Final stage
         FROM python:3.12-slim-bookworm

         # Copy the application from the builder
         COPY --from=builder --chown=app:app /app /app

         # Place executables in the environment at the front of the path
         ENV PATH="/app/.venv/bin:$PATH"

         # Set the working directory
         WORKDIR /app

ouput from :  src/train.py
<img width="1075" alt="image" src="https://github.com/user-attachments/assets/6a844b1a-9553-4c19-8a00-9d6241ace6ec">
ouput from :  src/eval.py
<img width="1262" alt="image" src="https://github.com/user-attachments/assets/b56d2f54-71e9-4ce5-9a76-60aef51cfa10">
ouput from :  src/infer.py
<img width="1615" alt="image" src="https://github.com/user-attachments/assets/95dae635-e28f-4da6-afb5-6ceebc0e90bf">



<h3>Prediction Results</h3>

  The model prediction gets saved in the predicted_images folder in the model artifacts.


  <table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/92cb4dd8-8f37-4ae5-a906-7447554720a9" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/2cde3ce2-3596-4bf9-bb88-9d2c996407eb" width="500"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c2a4bc37-078d-49d2-afbe-65cc0ccd3a82" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/7ca0ca0e-3295-49eb-a266-e2009eca271e" width="500"/></td>
  </tr>
   <tr>
    <td><img src="https://github.com/user-attachments/assets/97d12ed1-9f40-4d9e-929f-d282be82e882" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/3adf2fc0-2753-4cb9-bd93-b4bbcb236077" width="500"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c6bb007a-d7be-4942-8b04-2d8148e57522" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/2d96e61d-b49b-41cb-b342-871f2a43476d" width="500"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/20e0d992-23dc-4816-a6d1-d01efd04b0bb" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/c5ebc356-60f7-4ab1-bcca-22cfc5e281de" width="500"/></td>
  </tr>
</table>


<h3>Requirements</h3>

         Docker
         Kaggle API (for downloading the dataset)
         GitHub Codespaces or Visual Studio Code with the Remote Containers extension (for DevContainer setup)

  
    

//hello

##add me 
