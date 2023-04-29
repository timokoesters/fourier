use fft2d::slice::{fft_2d, fftshift, ifft_2d, ifftshift};
use image::GenericImageView;
use rustfft::num_complex::Complex;
use wasm_bindgen::prelude::*;

use log::warn;

use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle, WebDisplayHandle,
    WebWindowHandle,
};

struct WebWindow;
unsafe impl HasRawDisplayHandle for WebWindow {
    fn raw_display_handle(&self) -> RawDisplayHandle {
        RawDisplayHandle::Web(WebDisplayHandle::empty())
    }
}
unsafe impl HasRawWindowHandle for WebWindow {
    fn raw_window_handle(&self) -> RawWindowHandle {
        RawWindowHandle::Web(WebWindowHandle::empty())
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: wgpu::Texture,
    mode: Mode,
    mousedown: bool,
    last_mousepos: Option<(u32, u32)>,
}

enum Mode {
    Image {
        image: image::RgbaImage,
    },
    Fft {
        data: (Vec<Complex<f64>>, Vec<Complex<f64>>, Vec<Complex<f64>>),
    },
}

#[derive(Debug)]
enum CanvasEvent {
    MouseMove(u32, u32),
    MouseDown,
    MouseUp,
}

impl State {
    async fn new(canvas: &web_sys::HtmlCanvasElement, fft: bool) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        let surface = unsafe { instance.create_surface_from_canvas(&canvas) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, mut queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.describe().srgb)
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            width: 1024,
            height: 1024,
        };

        surface.configure(&device, &config);

        // Load image
        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_resized =
            diffuse_image.resize_exact(1024, 1024, image::imageops::FilterType::Lanczos3);

        let diffuse_rgba = diffuse_resized.to_rgba8();

        // Create texture
        let dimensions = diffuse_resized.dimensions();
        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            size: texture_size,
            mip_level_count: 1, // We'll talk about this a little later
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Most images are stored using sRGB so we need to reflect that here.
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("diffuse_texture"),
            // This is the same as with the SurfaceConfig. It
            // specifies what texture formats can be used to
            // create TextureViews for this texture. The base
            // texture format (Rgba8UnormSrgb in this case) is
            // always supported. Note that using a different
            // texture format is not supported on the WebGL2
            // backend.
            view_formats: &[],
        });

        // Create sampler
        let diffuse_texture_view =
            diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        // Create pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self::upload_image(&mut queue, &diffuse_texture, &diffuse_rgba);

        let mode = if fft {
            Mode::Fft {
                data: (Vec::new(), Vec::new(), Vec::new()),
            }
        } else {
            Mode::Image {
                image: diffuse_rgba,
            }
        };
        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            diffuse_bind_group,
            diffuse_texture,
            mode,
            mousedown: false,
            last_mousepos: None,
        }
    }

    fn upload_image(
        queue: &mut wgpu::Queue,
        diffuse_texture: &wgpu::Texture,
        image: &image::RgbaImage,
    ) {
        // Upload texture
        let dimensions = image.dimensions();
        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // The actual pixel data
            &image,
            // The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
                rows_per_image: std::num::NonZeroU32::new(dimensions.1),
            },
            texture_size,
        );
    }

    fn input(&mut self, event: &CanvasEvent, other: &mut State) -> bool {
        warn!("{event:?}");
        match event {
            CanvasEvent::MouseDown => {
                self.mousedown = true;
            }
            CanvasEvent::MouseUp => {
                self.mousedown = false;
                self.update(other);
            }
            CanvasEvent::MouseMove(x, y) => {
                let old_mousepos = self.last_mousepos;
                self.last_mousepos = Some((*x, *y));
                if !self.mousedown || old_mousepos.is_none() {
                    return false;
                }

                let old_x = old_mousepos.unwrap().0 as i32;
                let old_y = old_mousepos.unwrap().1 as i32;

                match &mut self.mode {
                    Mode::Image { image } => {
                        imageproc::drawing::draw_antialiased_line_segment_mut(
                            image,
                            (old_x, old_y),
                            (*x as i32, *y as i32),
                            [0, 0, 0, 255].into(),
                            imageproc::pixelops::interpolate,
                        );
                        Self::upload_image(&mut self.queue, &self.diffuse_texture, image);
                    }
                    Mode::Fft { data } => {
                        for ty in y.saturating_sub(50)..(y + 50).min(1024) {
                            for tx in x.saturating_sub(50)..(x + 50).min(1024) {
                                for c in [&mut data.0, &mut data.1, &mut data.2] {
                                    let delta = 1.0 - 30.0 / c[(1024 * ty + tx) as usize].norm();
                                    c[(1024 * ty + tx) as usize] *= delta.max(0.0);
                                    c[(1024 * 1024 - 1024 * ty - tx) as usize] *= delta.max(0.0);
                                }
                            }
                        }
                        Self::upload_image(
                            &mut self.queue,
                            &self.diffuse_texture,
                            &Self::image_from_complex(data),
                        );
                    }
                }
                self.render();
            }
            _ => {}
        }
        false
    }

    fn image_from_complex(
        data: &(Vec<Complex<f64>>, Vec<Complex<f64>>, Vec<Complex<f64>>),
    ) -> image::RgbaImage {
        let mut r_buffer = data.0.clone();
        let mut g_buffer = data.1.clone();
        let mut b_buffer = data.2.clone();

        let mut r_max = 0.0;
        for r in &mut r_buffer {
            let ln = r.norm().ln();
            *r = Complex { re: ln, im: 0.0 };
            if ln > r_max {
                r_max = ln;
            }
        }

        let mut g_max = 0.0;
        for g in &mut g_buffer {
            let ln = g.norm().ln();
            *g = Complex { re: ln, im: 0.0 };
            if ln > g_max {
                g_max = ln;
            }
        }

        let mut b_max = 0.0;
        for b in &mut b_buffer {
            let ln = b.norm().ln();
            *b = Complex { re: ln, im: 0.0 };
            if ln > b_max {
                b_max = ln;
            }
        }

        image::RgbaImage::from_fn(1024, 1024, |x, y| {
            let r = (r_buffer[(x + y * 1024) as usize].re / r_max * 255.0) as u8;
            let g = (g_buffer[(x + y * 1024) as usize].re / g_max * 255.0) as u8;
            let b = (b_buffer[(x + y * 1024) as usize].re / b_max * 255.0) as u8;
            image::Rgba::from([r, g, b, 255])
        })
    }

    fn update(&mut self, other: &mut State) {
        match (&self.mode, &mut other.mode) {
            (Mode::Image { image }, Mode::Fft { data }) => {
                let mut r_buffer: Vec<Complex<f64>> = image
                    .pixels()
                    .map(|&pix| Complex::new(pix[0] as f64 / 255.0, 0.0))
                    .collect();
                fft_2d(1024, 1024, &mut r_buffer);
                r_buffer = fftshift(1024, 1024, &r_buffer);
                data.0 = r_buffer;

                let mut g_buffer: Vec<Complex<f64>> = image
                    .pixels()
                    .map(|&pix| Complex::new(pix[1] as f64 / 255.0, 0.0))
                    .collect();
                fft_2d(1024, 1024, &mut g_buffer);
                g_buffer = fftshift(1024, 1024, &g_buffer);
                data.1 = g_buffer;

                let mut b_buffer: Vec<Complex<f64>> = image
                    .pixels()
                    .map(|&pix| Complex::new(pix[2] as f64 / 255.0, 0.0))
                    .collect();
                fft_2d(1024, 1024, &mut b_buffer);
                b_buffer = fftshift(1024, 1024, &b_buffer);
                data.2 = b_buffer;

                let image = Self::image_from_complex(&data);
                Self::upload_image(&mut other.queue, &other.diffuse_texture, &image);
                other.render();
            }
            (Mode::Fft { data }, Mode::Image { image }) => {
                let mut r_buffer = ifftshift(1024, 1024, &data.0);
                ifft_2d(1024, 1024, &mut r_buffer);
                let mut g_buffer = ifftshift(1024, 1024, &data.1);
                ifft_2d(1024, 1024, &mut g_buffer);
                let mut b_buffer = ifftshift(1024, 1024, &data.2);
                ifft_2d(1024, 1024, &mut b_buffer);

                *image = image::RgbaImage::from_fn(1024, 1024, |x, y| {
                    let r = (r_buffer[(x + y * 1024) as usize].re / 1024.0 / 1024.0 * 255.0) as u8;
                    let g = (g_buffer[(x + y * 1024) as usize].re / 1024.0 / 1024.0 * 255.0) as u8;
                    let b = (b_buffer[(x + y * 1024) as usize].re / 1024.0 / 1024.0 * 255.0) as u8;
                    image::Rgba::from([r, g, b, 255])
                });

                Self::upload_image(&mut other.queue, &other.diffuse_texture, &image);
                other.render();
            }
            _ => {}
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");

    let doc = web_sys::window().unwrap().document().unwrap();
    let canvas_before = doc.get_element_by_id("canvas-before").unwrap();
    let canvas_after = doc.get_element_by_id("canvas-after").unwrap();

    let canvas_before: &'static _ = Box::leak(Box::new(
        canvas_before
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap(),
    ));
    let canvas_after: &'static _ = Box::leak(Box::new(
        canvas_after
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap(),
    ));

    canvas_before.set_width(1024);
    canvas_before.set_height(1024);
    canvas_after.set_width(1024);
    canvas_after.set_height(1024);

    let mut state_before = State::new(&canvas_before, false).await;
    let mut state_after = State::new(&canvas_after, true).await;

    let mut receiver_before = setup_listeners(&canvas_before);
    let mut receiver_after = setup_listeners(&canvas_after);

    state_before.render();
    state_before.update(&mut state_after);

    loop {
        tokio::select! {

        Some(event) = receiver_before.recv() => {state_before.input(&event, &mut state_after);}
        Some(event) = receiver_after.recv() => {state_after.input(&event, &mut state_before);}        }
    }

    //while let Some(event) = .await {
    //;
    //}
}

fn setup_listeners(
    canvas: &'static web_sys::HtmlCanvasElement,
) -> tokio::sync::mpsc::UnboundedReceiver<CanvasEvent> {
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();

    let sender2 = sender.clone();
    {
        let closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
            let rect = canvas.get_bounding_client_rect();
            let width = canvas.width() as f32;
            let height = canvas.height() as f32;
            let x = event.offset_x() as f32 * (width / rect.width() as f32);
            let y = event.offset_y() as f32 * (height / rect.height() as f32);
            sender2.send(CanvasEvent::MouseMove(x as u32, y as u32));
        }) as Box<dyn FnMut(_)>);

        canvas
            .add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    let sender2 = sender.clone();
    {
        let closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
            sender2.send(CanvasEvent::MouseDown);
        }) as Box<dyn FnMut(_)>);

        canvas
            .add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    let sender2 = sender.clone();
    {
        let closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
            sender2.send(CanvasEvent::MouseUp);
        }) as Box<dyn FnMut(_)>);

        canvas
            .add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    receiver
}
