#![cfg(feature = "trace")]

use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{fmt::{self, format::FmtSpan}, prelude::*};

/// Initialises standard output and Chrome tracing.
///
/// Keep the returned guard alive until tracing is finished.
/// Dropping the guard flushes the Chrome trace to `trace.json`.
pub fn init_tracing() -> impl Drop {
    // The ChromeLayer writes to the file on Drop therefore a guard is necessary.
    let (chrome_layer, guard) = ChromeLayerBuilder::new()
        .file("trace.json")
        .include_args(true)
        .build();

    let stdout_layer = fmt::layer()
        .with_span_events(FmtSpan::CLOSE);

    tracing_subscriber::registry()
        .with(chrome_layer)
        .with(stdout_layer)
        .init();

    guard
}