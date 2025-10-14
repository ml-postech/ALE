from functools import partial
import gradio as gr

from ale.edit import ale_edit, load_edit_pipeline

css = """.cycle-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.cycle-diffusion-div div h1{font-weight:900;margin-bottom:7px}.cycle-diffusion-div p{margin-bottom:10px;font-size:94%}.cycle-diffusion-div p a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}"""
intro = """
<div style="display: flex;align-items: center;justify-content: center">
    <h1 style="margin-left: 12px;text-align: center;margin-bottom: 7px;display: inline-block">ALE</h1>
    <h3 style="display: inline-block;margin-left: 10px;margin-top: 6px;font-weight: 500">Attribute-Leakage-Free Editing</h3>
</div>
"""

if __name__ == "__main__":
    pipe = load_edit_pipeline()

    with gr.Blocks(css=css) as demo:
        gr.HTML(intro)
        with gr.Row():
            with gr.Column(scale=55):
                with gr.Group():
                    img = gr.Image(
                        label="Input image", height=512, width=512, type="pil"
                    )
                    image_out = gr.Image(
                        label="Output image", height=512, width=512, type="pil"
                    )

            with gr.Column(scale=45):
                with gr.Row():
                    button_edit = gr.Button(value="Run")
                with gr.Tab("ALE Options"):
                    with gr.Group():
                        # with gr.Row():
                        #     button_edit = gr.Button(value="Run")
                        with gr.Row():
                            source_prompt = gr.Textbox(
                                label="Source prompt",
                                placeholder="Source prompt describes the input image",
                            )
                            target_prompt = gr.Textbox(
                                label="Target prompt",
                                placeholder="Target prompt describes the output image",
                            )
                        with gr.Row():
                            source_prompt2 = gr.Textbox(
                                label="Source prompt2",
                                placeholder="Source prompt describes the input image",
                            )
                            target_prompt2 = gr.Textbox(
                                label="Target prompt2",
                                placeholder="Target prompt describes the output image",
                            )
                        with gr.Row():
                            source_prompt3 = gr.Textbox(
                                label="Source prompt3",
                                placeholder="Source prompt describes the input image",
                            )
                            target_prompt3 = gr.Textbox(
                                label="Target prompt3",
                                placeholder="Target prompt describes the output image",
                            )
                        with gr.Row():
                            negative_prompt = gr.Textbox(
                                label="Negative prompt", placeholder=""
                            )
                        with gr.Row():
                            self_replace_steps = gr.Slider(
                                label="Self attention injection schedule",
                                value=0.5,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                            )
                            dilation_percent = gr.Slider(
                                label="Mask Dilation Ratio",
                                value=0.04,
                                minimum=0.0,
                                maximum=0.1,
                                step=0.005,
                            )

                with gr.Tab("Advanced Options"):
                    with gr.Group():
                        with gr.Row():
                            num_inference_steps = gr.Slider(
                                label="Inference steps",
                                value=15,
                                minimum=1,
                                maximum=50,
                                step=1,
                            )
                        with gr.Row():
                            width = gr.Slider(
                                label="Width",
                                value=768,
                                minimum=512,
                                maximum=1024,
                                step=1,
                            )
                            height = gr.Slider(
                                label="Height",
                                value=768,
                                minimum=512,
                                maximum=1024,
                                step=1,
                            )
                        with gr.Row():
                            seed = gr.Slider(
                                0, 2147483647, label="Seed", value=0, step=1
                            )
                        with gr.Row():
                            box_threshold = gr.Slider(
                                label="SAM box thresh",
                                value=0.45,
                                minimum=0,
                                maximum=1,
                            )
                            text_threshold = gr.Slider(
                                label="SAM text thresh", value=0.4, minimum=0, maximum=1
                            )
                        with gr.Row():
                            guidance_s = gr.Slider(
                                label="Source guidance scale",
                                value=1,
                                minimum=1,
                                maximum=10,
                            )
                            guidance_t = gr.Slider(
                                label="Target guidance scale",
                                value=2,
                                minimum=1,
                                maximum=10,
                            )
        edit_inputs = [
            img,
            source_prompt,
            target_prompt,
            source_prompt2,
            target_prompt2,
            source_prompt3,
            target_prompt3,
            negative_prompt,
            guidance_s,
            guidance_t,
            num_inference_steps,
            width,
            height,
            seed,
            self_replace_steps,
            dilation_percent,
            box_threshold,
            text_threshold,
        ]

        edit_func = partial(ale_edit, pipe=pipe)

        button_edit.click(
            edit_func,
            inputs=edit_inputs,
            outputs=image_out,
        )

    demo.launch(debug=False, share=False)
