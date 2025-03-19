use image::GenericImageView;
use ndarray::{Array3, Axis, CowArray};
use ort::{Environment, SessionBuilder, Value};
use std::{error::Error, sync::Arc};

// 图像预处理函数
fn preprocess_image(image_path: &str) -> Result<Array3<f32>, Box<dyn Error>> {
    let img = image::open(image_path)?;
    // 调整图像大小为 112x112
    let resized = img.resize_exact(112, 112, image::imageops::FilterType::Nearest);

    let mut input = Array3::zeros((3, 112, 112));

    for (x, y, pixel) in resized.pixels() {
        input[[0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
        input[[1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
        input[[2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
    }

    // 归一化
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for c in 0..3 {
        for y in 0..112 {
            for x in 0..112 {
                input[[c, y, x]] = (input[[c, y, x]] - mean[c]) / std[c];
            }
        }
    }

    Ok(input)
}

// 运行推理函数
fn run_inference(session: &ort::Session, input: Array3<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
    let input_data = input.insert_axis(Axis(0));
    let binding = CowArray::from(input_data).into_dyn();
    let input_tensor = Value::from_array(session.allocator(), &binding)?;
    let outputs = session.run(vec![input_tensor])?;
    let output_tensor = outputs[0].try_extract::<f32>()?;
    let output_array = output_tensor.view();
    Ok(output_array.iter().cloned().collect())
}

// 加载标签文件
fn load_labels(file_path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let content = std::fs::read_to_string(file_path)?;
    let labels = content.lines().map(|line| line.to_string()).collect();
    Ok(labels)
}

fn main() -> Result<(), Box<dyn Error>> {
    // 初始化 ONNX Runtime 环境
    let environment = Arc::new(Environment::builder().with_name("resnet50_env").build()?);

    // 加载 ONNX 模型
    let session = SessionBuilder::new(&environment)?.with_model_from_file("resnet50-112.onnx")?;

    for i in 1..8 {
        let filename = format!("images/face{}.jpeg", i);
        // 图像预处理
        let input = preprocess_image(filename.as_str())?;

        // 运行推理
        let output = run_inference(&session, input)?;
        let output = softmax(&output);

        // 找到最大概率的类别索引
        let max_index = output
            .iter()
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // 加载标签文件
        let labels = load_labels("imagenet_classes.txt")?;

        // 输出分类结果
        println!(
            "Predicted class: {} {}",
            labels[max_index], output[max_index]
        );
    }
    Ok(())
}

fn softmax(input: &[f32]) -> Vec<f32> {
    // 找到输入向量中的最大值
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // 计算指数值
    let exp_values: Vec<f32> = input.iter().map(|x| (x - max_val).exp()).collect();

    // 计算指数值的总和
    let sum_exp = exp_values.iter().sum::<f32>();

    // 计算 Softmax 值
    exp_values.iter().map(|x| x / sum_exp).collect()
}
