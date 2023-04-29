#[tokio::main]
async fn main() {
    // build our application with a single route
    let app = axum::Router::new().nest("/", axum_static::static_router("."));

    println!("Running on http://0.0.0.0:3000/fourier.html");
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
