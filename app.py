"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import gradio as gr
from demo.gradio_func import (
    extract_openpose_keypoints,
    get_pose_from_example,
    generate_images,
    hallucinate,
    reconstruct,
    select_gen_images,
    get_select_index
)

scheduler_list = ['Default',
                  'DPM++ 2M', 'DPM++ 2M Karras', 
                  'DPM2', 'DPM2 Karras',
                  'DPM2 a', 'DPM2 a Karras',
                  'Euler', 'Euler Ancestral']      

with gr.Blocks(css = ".output-image, .input-image, .image-preview {height: 400px !important} ") as demo:

    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
            <h1 > SiTH: Single-view Textured Human Reconstruction with Image-Conditioned Diffusion </h1>
            <h5 style="margin: 0;">If you like our demo, please give us a star on Github to stay tuned with the latest updates.</h5>
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <a href='https://arxiv.org/abs/2311.15855'><img src='https://img.shields.io/badge/Arxiv-2311.15855-red'></a>
                    <a href='https://ait.ethz.ch/sith'><img src='https://img.shields.io/badge/Project_Page-SiTH-green' alt='Project Page'></a>
                    <a href='https://github.com/SiTH-Diffusion/SiTH'><img src='https://img.shields.io/badge/Github-SiTH-blue'></a>
                    </div>
            </div>
        </div>
        """)
    with gr.Row():
        with gr.Column():
            gr.HTML(
            """
            IMPORTNAT NOTICE: The results of this demo should not be used for
            evaluation purposes due to different hyperparameters and training data. 
            The inference time of this demo page does not reflect the actual inference time of the model. 
            For benchmark evaluation details, please refer to our offical github repo.
            """
            )

        with gr.Column():
            gr.HTML(
            """
            DISCLAIMER: The demo is provided for the creation of 3D human models in 
            compliance with ethical guidelines and legal standards. The authors are not 
            responsible for the misuse of these tools by users to create content that may 
            interfere with privacy or sensitive matters. Users are solely responsible for 
            ensuring that their creations with our demo do not violate any laws, interfere 
            with the privacy or rights of individuals, and are not distributed for 
            unauthorized purposes.
            """
            )

    gr.HTML(
        """
        <hr>
        """
    )

    with gr.Row():

        with gr.Column():
            gr.HTML(
            """
            <h2>Step 1: Select and adjust a pose</h2>
            """)
            pose_image_vis = gr.Image(type="pil", label="Pose Image")
            example_img = gr.Image(type="filepath", label="Pose Image",visible=False)
            
            pose_image = gr.State()
            image_cache = gr.State()
            json_cache = gr.State()
            V_2d = gr.State()
            V_3d = gr.State()
            tgt_uv = gr.State()
            vis = gr.State()

            with gr.Column():
                with gr.Accordion("Upload my own SMPL-X pose", open=False):
                    upload_button = gr.UploadButton(file_types=[".json"], label="Upload a JSON file")
                gr.HTML(
                """
                <h4>👇Select an pose below to start. You can find more poses and SMPL-X parameters in JSON format in our <a href='https://custom-humans.github.io/#download'>CustomHumans</a> dataset.</h4>
                """)
                example = gr.Examples(inputs=[
                                    example_img,
                                    ],
                                    examples_per_page=16,
                                    examples=[ 'data/gradio/pose_001.png',
                                             'data/gradio/pose_002.png', 
                                             'data/gradio/pose_003.png', 
                                             'data/gradio/pose_004.png',
                                             'data/gradio/pose_005.png',
                                             'data/gradio/pose_006.png',
                                             'data/gradio/pose_007.png',
                                             'data/gradio/pose_008.png',
                                             'data/gradio/pose_009.png',
                                             'data/gradio/pose_010.png',
                                             'data/gradio/pose_011.png',
                                             'data/gradio/pose_012.png',
                                             'data/gradio/pose_013.png',
                                             'data/gradio/pose_014.png',
                                             'data/gradio/pose_015.png',
                                             'data/gradio/pose_016.png'                                  
                                            ],
                                  fn=get_pose_from_example,
                                  outputs=[pose_image, pose_image_vis, V_2d, V_3d, tgt_uv, vis, json_cache],
                                  run_on_click=True
                                  )
            with gr.Accordion("Adjust body size and keypoint position", open=False):

                scale = gr.Slider(minimum=0.5, maximum=1.5, step=0.01, label="Scale", value=1.0)

                with gr.Row():
                    offset_x = gr.Slider(minimum=-0.5, maximum=0.5, step=0.01, label="Offset X", value=0.0)
                    offset_y = gr.Slider(minimum=-0.5, maximum=0.5, step=0.01, label="Offset Y", value=0.0)

                refresh_button = gr.Button(value="Refresh Pose")

            
            upload_button.upload(fn=extract_openpose_keypoints,
                                 inputs=[upload_button,
                                            scale,
                                            offset_x,
                                            offset_y
                                            ],
                                  outputs=[pose_image, pose_image_vis, V_2d, V_3d, tgt_uv, vis, json_cache],
                                    concurrency_limit=1
                                    )

            refresh_button.click(fn=extract_openpose_keypoints,
                                inputs=[json_cache,
                                        scale,
                                        offset_x,
                                        offset_y
                                        ], 
                                outputs=[pose_image, pose_image_vis, V_2d, V_3d, tgt_uv, vis, json_cache],
                                concurrency_limit=1)

        with gr.Column():

            gr.HTML(
            """
            <h2>Step 2: Generate and choose a image with ControlNet</h2>
            """)

            gen_image = gr.Gallery(label="Generated Images", columns=[2],rows=[2])
            gr.HTML(
                """
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <h4>👆Select one of the images above after you see the results.</h4>
                </div>
                """)
            pos_prompt = gr.Textbox(label="Positive prompt", value="male, jacket, shorts, bare hand, full-body, front view, plain background, sharp focus, detailed skin, raw photo")
            neg_prompt = gr.Textbox(label="Negative prompt", value="nsfw, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, holding objects")
                
            with gr.Accordion("More diffusion options", open=False):
                sampler_name = gr.Dropdown(label='Sampling method', choices=scheduler_list, value=scheduler_list[0])
                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling steps", value=50)

                with gr.Row():
                    num_images = gr.Slider(minimum=1, maximum=8, step=1, label="Batch size", value=4)
                    seed = gr.Number(2434, label='Seed')
                with gr.Row():
                    cfg_scale = gr.Slider(minimum=1.0, maximum=10.0, value=3.5, step=0.5, label="CFG scale")
                    cond_scale = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="ControlNet conditioning scale")

            controlnet_button = gr.Button(value="Run ControlNet")

            controlnet_button.click(fn=generate_images, 
                             inputs=[pose_image,
                                     pos_prompt,
                                     neg_prompt,
                                     sampler_name,
                                     steps,
                                     num_images,
                                     seed,
                                     cfg_scale,
                                     cond_scale
                                    ], 
                             outputs=[gen_image, image_cache],
                             concurrency_limit=1)

    with gr.Accordion("Check mask removal and pose alignment", open=False):

        gr.HTML(
            """
            <hr>
            """)
        
        with gr.Row():

            with gr.Column():
                ori_img = gr.Image(type="pil", label="Selected front image")
            with gr.Column():
                rgba_image = gr.Image(type="pil", image_mode ='RGBA', label="Background removal")
            with gr.Column():
                overlay_image = gr.Image(type="pil", label="Pose alignment")

    gr.HTML(
        """
        <hr>
        <h2>Step 3: Hallucinate and choose a back-view image</h2>
        """)

    with gr.Row():
        
        back_images = gr.Gallery(label="Generated Images", columns=[4],rows=[1])
        back_cache = gr.State()
    with gr.Row():
        gr.HTML(
            """
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <h4>👆Select one of the images above after you see the results.</h4>
            </div>
            """)
    with gr.Row():
        
        with gr.Accordion("More hallucination options", open=False):

            with gr.Column():
                h_sampler_name = gr.Dropdown(label='Sampling method', choices=scheduler_list, value=scheduler_list[0])
                h_steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling steps", value=50)
            with gr.Column():
                with gr.Row():
                    h_num_images = gr.Slider(minimum=1, maximum=8, step=1, label="Batch size", value=4)
                    h_seed = gr.Number(2434, label='seed')
                with gr.Row():
                    h_cfg_scale = gr.Slider(minimum=1.0, maximum=10.0, value=3.5, step=0.5, label="CFG scale")
                    h_cond_scale = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="ControlNet conditioning scale")

        with gr.Column():
            hallucinate_button = gr.Button(value="Run Hallucination")
        
        hallucinate_button.click(fn=hallucinate, 
                             inputs=[rgba_image,
                                     tgt_uv,
                                     h_sampler_name,
                                     h_steps,
                                     h_num_images,
                                     h_seed,
                                     h_cfg_scale,
                                     h_cond_scale
                                    ], 
                             outputs=[back_images, back_cache],
                             concurrency_limit=1)
    gr.HTML(
        """
        <hr>
        <h2>Step 4: Reconstruct a 3D textured mesh</h2>
        """)

    with gr.Row():
        output_model = gr.Model3D()

    
    with gr.Row():
        with gr.Accordion("More reconstruction options", open=False):

            bg_color = gr.ColorPicker(label="Background color fill", value="#000000")
            iters = gr.Number(1, label='Erorsion Iteration')

        with gr.Column():
            recon_bottom = gr.Button(value="Reconstruct texture mesh")

    with gr.Accordion("Check selected images", open=False):

        with gr.Row():
            with gr.Column():
                select_front = gr.Image(type="pil", image_mode ='RGBA', label="Selected front image")
            with gr.Column():
                select_back = gr.Image(type="pil", label="Selected back image")


            gen_image.select(
                fn=select_gen_images,
                inputs=[image_cache, V_2d],
                outputs=[ori_img, rgba_image, select_front, overlay_image],

            )
            back_images.select(
                fn=get_select_index,
                inputs=[back_cache],
                outputs=[select_back],
            )



        recon_bottom.click(fn=reconstruct, 
                            inputs=[select_front,
                                     select_back,
                                     V_3d,
                                     vis,
                                     bg_color,
                                     iters
                                    ], 
                             outputs=[output_model, select_front, select_back],
                             concurrency_limit=1)


if __name__ == "__main__":
    demo.queue(max_size=4)
    demo.launch()