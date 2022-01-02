import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.*;
import javafx.scene.*;
import javafx.scene.canvas.*;
import javafx.scene.control.*;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.*;
import javafx.scene.text.Text;
import javafx.stage.*;

public class SignalFreq extends Application {
  final int IMAGEW = 512;
  final int IMAGEH = 512;
  final TwoDimData originalimage, grayimage, freqimage;
  final TwoDimData kernelimage0, kernelimage1, kernelimage2;
  final TwoDimData maskimage0, maskimage1, maskimage2;
  static BufferedImage dummykernelone, dummykernelzero;
  static BufferedImage dummymask;

  Button initialButton;
  Button resetButton;

  Canvas canvasi;
  GraphicsContext gci;
  WritableImage imagei;
  Slider gaussl;
  Slider gaussh;
  Slider amplitude;
  Slider interval;
  ToggleButton ifilterButton;
  ToggleButton biasedIfilterButton;

  Button transformButton;
  Button invtransformButton;

  Canvas canvasf;
  GraphicsContext gcf;
  WritableImage imagef;
  Slider freqf0;
  Slider freqf1;
  ToggleButton oneminusButton;
  ToggleButton oneButton;
  ToggleButton ffilterButton;

  {
    dummykernelone = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
    int constv = (255 << 16) + (255 << 8) + 255;
    dummykernelone.setRGB(0, 0, constv);
    dummykernelzero = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
    dummykernelzero.setRGB(0, 0, 0);
    dummymask = new BufferedImage(IMAGEW, IMAGEH, BufferedImage.TYPE_INT_RGB);
    for (int y = 0; y < IMAGEH; y++) {
      for (int x = 0; x < IMAGEW; x++) {
        dummymask.setRGB(x, y, constv);
      }
    }
  }

  public SignalFreq() {
    originalimage = new TwoDimData(IMAGEW, IMAGEH);
    grayimage = new TwoDimData(IMAGEW, IMAGEH);
    freqimage = new TwoDimData(IMAGEW, IMAGEH);
    kernelimage0 = new TwoDimData(1, 1);
    kernelimage1 = new TwoDimData(1, 1);
    kernelimage2 = new TwoDimData(1, 1);
    maskimage0 = new TwoDimData(IMAGEW, IMAGEH);
    maskimage1 = new TwoDimData(IMAGEW, IMAGEH);
    maskimage2 = new TwoDimData(IMAGEW, IMAGEH);
    kernelimage0.setDefault(dummykernelone);
    kernelimage1.setDefault(dummykernelzero);
    kernelimage2.setDefault(dummykernelone);
    maskimage0.setDefault(dummymask, dummymask);
    maskimage1.setDefault(dummymask, dummymask);
    maskimage2.setDefault(dummymask, dummymask);
    grayimage.setDefault(dummymask);
    freqimage.setDefault(dummymask, dummymask);
  }

  public void start(Stage stage) {
    FileChooser chooser = new FileChooser();

    initialButton = new Button("Set Image");
    resetButton = new Button("reset");

    canvasi = new Canvas(IMAGEW, IMAGEH);
    gci = canvasi.getGraphicsContext2D();
    imagei = new WritableImage(IMAGEW, IMAGEH);
    gaussl = new Slider(0.1, 10, 0.1); // range is 1 to 101 init value is 1
    gaussl.setPrefWidth(IMAGEW - 30);
    gaussh = new Slider(0, 10, 0);
    gaussh.setPrefWidth(IMAGEW - 30);
    amplitude = new Slider(1 / 16.0, 128, 1.0);
    amplitude.setPrefWidth(IMAGEW - 30);
    interval = new Slider(1, 16, 1);
    interval.setPrefWidth(IMAGEW - 133);
    ifilterButton = new ToggleButton("Conv. Kernel File");
    biasedIfilterButton = new ToggleButton("Biased Conv. Kernel File");
    ToggleGroup tgs = new ToggleGroup();
    ifilterButton.setToggleGroup(tgs);
    biasedIfilterButton.setToggleGroup(tgs);

    transformButton = new Button("Transform");
    invtransformButton = new Button("InversTrans.");

    canvasf = new Canvas(IMAGEW, IMAGEH);
    gcf = canvasf.getGraphicsContext2D();
    imagef = new WritableImage(IMAGEW, IMAGEH);
    freqf0 = new Slider(0, IMAGEW * 2, 0);
    freqf0.setPrefWidth(IMAGEW - 30);
    freqf1 = new Slider(0, IMAGEW * 2, IMAGEW * 2);
    freqf1.setPrefWidth(IMAGEW - 30);
    oneminusButton = new ToggleButton("(1.-k.*l.*m).*o");
    oneButton = new ToggleButton("(k.*l.*m).*o");
    ToggleGroup tgf = new ToggleGroup();
    oneButton.setToggleGroup(tgf);
    oneButton.setSelected(true);
    oneminusButton.setToggleGroup(tgf);
    ffilterButton = new ToggleButton("Freq Mask File");

    VBox imageside = new VBox();
    VBox freqside = new VBox();

    VBox trans = new VBox();
    HBox ifilbox = new HBox();
    ifilbox.setAlignment(Pos.CENTER);
    ifilbox.getChildren().addAll(biasedIfilterButton, ifilterButton);
    imageside.getChildren().addAll(canvasi, new HBox(new Text("f: "), gaussl),
        new HBox(new Text("g: "), gaussh), new HBox(new Text("h: "), ifilbox),
        new HBox(new Text("a: "), amplitude), new Text("a.*(conv(h,(conv(f,s)-conv(g,s))))"));
    trans.getChildren().addAll(transformButton, invtransformButton);
    freqside.getChildren().addAll(canvasf, new HBox(new Text("k: "), freqf0),
        new HBox(new Text("l: "), freqf1), new HBox(new Text("m: "), ffilterButton),
        new HBox(oneButton, oneminusButton));
    imageside.setAlignment(Pos.CENTER);
    freqside.setAlignment(Pos.CENTER);
    trans.setAlignment(Pos.CENTER);

    HBox tophb = new HBox();
    VBox vb = new VBox();
    HBox hb = new HBox();
    tophb.getChildren().addAll(initialButton, resetButton);
    vb.getChildren().addAll(tophb, hb, new HBox(new Text("sampling interval: "), interval));
    vb.setAlignment(Pos.CENTER);
    hb.getChildren().addAll(imageside, trans, freqside);

    Scene scene = new Scene(vb, 1150, 660);

    initialButton.setOnAction(e -> {
      chooser.setTitle("Load Image");
      File file = chooser.showOpenDialog(stage);
      if (file != null) {
        grayimage.setDefault(file);
        originalimage.setDefault(grayimage);
        reset();
      }
    });

    resetButton.setOnAction(e -> { reset(); });

    gaussl.setOnMouseReleased(ev -> {
      double sigma = gaussl.getValue();
      BufferedImage kernel = SignalUtil.createGaussianImage(sigma, sigma);
      kernelimage0.setDefault(kernel);
      kernelimage0.normalizeCurrentL1();
      updateSigalImage();
    });

    gaussl.setOnMouseDragged(ev -> {
      double sigma = gaussl.getValue();
      BufferedImage kernel = SignalUtil.createGaussianImage(sigma, sigma);
      SwingFXUtils.toFXImage(kernel, imagei);
      gci.drawImage(
          imagei, (IMAGEW - kernel.getWidth() + 1) / 2, (IMAGEH - kernel.getHeight() + 1) / 2);
    });

    gaussh.setOnMouseReleased(ev -> {
      double sigma = gaussh.getValue();
      BufferedImage kernel = SignalUtil.createGaussianImage(sigma, sigma);
      kernelimage1.setDefault(kernel);
      if (sigma != 0.0) {
        kernelimage1.normalizeCurrentL1();
      }
      updateSigalImage();
    });

    gaussh.setOnMouseDragged(ev -> {
      double sigma = gaussh.getValue();
      BufferedImage kernel = SignalUtil.createGaussianImage(sigma, sigma);
      SwingFXUtils.toFXImage(kernel, imagei);
      gci.drawImage(
          imagei, (IMAGEW - kernel.getWidth() + 1) / 2, (IMAGEH - kernel.getHeight() + 1) / 2);
    });

    ifilterButton.setOnAction(e -> {
      chooser.setTitle("Conv. Kernel File");
      File file = chooser.showOpenDialog(stage);
      if (file != null) {
        kernelimage2.setResizeAndDefault(file);
        kernelimage2.normalizeCurrentL1();
        kernelimage2.printCurrentReal();
      } else {
        kernelimage2.setDefault(dummykernelone);
        ifilterButton.setSelected(false);
      }
      updateSigalImage();
    });

    biasedIfilterButton.setOnAction(e -> {
      chooser.setTitle("Biased Conv. Kernel File");
      File file = chooser.showOpenDialog(stage);
      if (file != null) {
        kernelimage2.setResizeAndDefault(file);
        kernelimage2.subtractWith(128.0 / 255.0, 0);
        kernelimage2.normalizeCurrentL2();
        kernelimage2.printCurrentReal();
      } else {
        kernelimage2.setDefault(dummykernelone);
        biasedIfilterButton.setSelected(false);
      }
      updateSigalImage();
    });

    amplitude.setOnMouseReleased(ev -> { updateSigalImage(); });

    interval.setOnMouseDragged(ev -> { updateSigalImage(); });

    interval.setOnMouseReleased(ev -> { updateSigalImage(); });

    transformButton.setOnAction(e -> {
      SignalUtil.FourierTransform(grayimage, freqimage);
      resetFrequencyImage();
    });

    invtransformButton.setOnAction(e -> {
      SignalUtil.InverseFourierTransform(freqimage, grayimage);
      resetSignalImage();
    });

    freqf0.setOnMouseReleased(ev -> {
      int val = (int) freqf0.getValue();
      BufferedImage mask = SignalUtil.createGaussianMaskImage(IMAGEW, IMAGEH, val, val);
      maskimage0.setDefault(mask, mask);
      updateFrequencyImage();
    });

    freqf0.setOnMouseDragged(ev -> {
      int val = (int) freqf0.getValue();
      BufferedImage mask = SignalUtil.createGaussianMaskImage(IMAGEW, IMAGEH, val, val);
      maskimage0.setDefault(mask, mask);
      showMask();
    });

    freqf1.setOnMouseReleased(ev -> {
      int val = (int) freqf1.getValue();
      BufferedImage mask = SignalUtil.createGaussianMaskImage(IMAGEW, IMAGEH, val, val);
      maskimage1.setDefault(mask, mask);
      maskimage1.multiplyWith(-1.0);
      maskimage1.addWith(1.0, 1.0);
      updateFrequencyImage();
    });

    freqf1.setOnMouseDragged(ev -> {
      int val = (int) freqf1.getValue();
      BufferedImage mask = SignalUtil.createGaussianMaskImage(IMAGEW, IMAGEH, val, val);
      maskimage1.setDefault(mask, mask);
      maskimage1.multiplyWith(-1.0);
      maskimage1.addWith(1.0, 1.0);
      showMask();
    });

    ffilterButton.setOnAction(e -> {
      chooser.setTitle("Freq Mask File");
      File file = chooser.showOpenDialog(stage);
      if (file != null) {
        maskimage2.setDefault(file, file);
        ffilterButton.setSelected(true);
      } else {
        maskimage2.setDefault(dummymask, dummymask);
        ffilterButton.setSelected(false);
      }
      updateFrequencyImage();
    });

    oneminusButton.setOnMousePressed(e -> {
      oneminusButton.setSelected(true);
      showMask();
    });

    oneminusButton.setOnMouseReleased(e -> { updateFrequencyImage(); });

    oneminusButton.setOnAction(e -> { oneminusButton.setSelected(!oneminusButton.isSelected()); });

    oneButton.setOnMousePressed(e -> {
      oneButton.setSelected(true);
      oneminusButton.setSelected(false);
      showMask();
    });

    oneButton.setOnMouseReleased(e -> { updateFrequencyImage(); });

    oneButton.setOnAction(e -> { oneButton.setSelected(!oneminusButton.isSelected()); });

    stage.setTitle("Relationship between Signal and Freqency");
    stage.setScene(scene);
    stage.show();
    updateSigalImage();
    updateFrequencyImage();
  }

  private void reset() {
    grayimage.setDefault(originalimage);
    freqimage.setDefault(dummymask, dummymask);
    interval.setValue(1);
    resetSignalImage();
    resetFrequencyImage();
  }

  private void resetSignalImage() {
    SwingFXUtils.toFXImage(grayimage.exportBiasedBufferedImage(), imagei);
    gci.drawImage(imagei, 0, 0);
    gaussl.setValue(0.1);
    gaussh.setValue(0);
    kernelimage0.setDefault(dummykernelone);
    kernelimage1.setDefault(dummykernelzero);
    kernelimage2.setDefault(dummykernelone);
    ifilterButton.setSelected(false);
    biasedIfilterButton.setSelected(false);
    amplitude.setValue(1);
  }

  private void resetFrequencyImage() {
    SwingFXUtils.toFXImage(freqimage.exportMagnitudeLoggedBufferedImage(), imagef);
    gcf.drawImage(imagef, 0, 0);
    oneButton.setSelected(true);
    freqf0.setValue(0);
    freqf1.setValue(IMAGEW * 2);
    ffilterButton.setSelected(false);
    maskimage0.setDefault(dummymask, dummymask);
    maskimage1.setDefault(dummymask, dummymask);
    maskimage2.setDefault(dummymask, dummymask);
  }

  private void updateSigalImage() {
    grayimage.reset();
    TwoDimData tmp0 = grayimage.convolutionOut(kernelimage1);
    grayimage.convolutionWith(kernelimage0);
    grayimage.subtractWith(tmp0);
    grayimage.convolutionWith(kernelimage2);
    grayimage.multiplyWith(amplitude.getValue());

    int intervalv = (int) interval.getValue();
    System.out.println("interval: " + intervalv);
    BufferedImage img = grayimage.exportBiasedBufferedImage();
    Graphics2D gc = img.createGraphics();
    for (int y = 0; y < IMAGEH - intervalv; y = y + intervalv) {
      for (int x = 0; x < IMAGEW - intervalv; x = x + intervalv) {
        gc.setColor(new Color(img.getRGB(x + intervalv / 2, y + intervalv / 2)));
        gc.fillRect(x, y, intervalv, intervalv);
      }
    }
    SwingFXUtils.toFXImage(img, imagei);
    gci.drawImage(imagei, 0, 0);
  }

  private void updateFrequencyImage() {
    freqimage.reset();
    TwoDimData tmp = maskimage0.multiplyOut(maskimage1);
    tmp.multiplyWith(maskimage2);
    if (oneminusButton.isSelected()) {
      tmp.multiplyWith(-1.0);
      tmp.addWith(1.0, 1.0);
    }
    freqimage.multiplyWith(tmp);
    SwingFXUtils.toFXImage(freqimage.exportMagnitudeLoggedBufferedImage(), imagef);
    gcf.drawImage(imagef, 0, 0);
  }

  private void showMask() {
    TwoDimData tmp = maskimage0.multiplyOut(maskimage1);
    tmp.multiplyWith(maskimage2);
    if (oneminusButton.isSelected()) {
      tmp.multiplyWith(-1.0);
      tmp.addWith(1.0, 1.0);
    }
    SwingFXUtils.toFXImage(tmp.exportBufferedImage(), imagef);
    gcf.drawImage(imagef, 0, 0);
  }

  public static void main(String[] args) {
    launch(args);
  }
}
