import streamlit as st
import cv2
import statistics
import tempfile
import os

def analyze_video_worker_time(video_path, min_area_threshold=5000):
    """
    Belirtilen video dosyasında bir kişinin/hareketin varlığını
    basit bir hareket algılama ile tespit ederek,
    toplam aktif süre (saniye) ve istatistiksel bilgileri döndürür.

    Parametreler
    -----------
    video_path : str
        Video dosya yolu
    min_area_threshold : int
        'Hareketli piksel' alanı için minimum eşik değeri
        (karedeki hareket alanı bu eşiğin üzerindeyse o kare aktif sayılır).

    Dönüş
    -----
    dict:
        {
          "fps": video fps değeri,
          "frame_count": toplam kare sayısı,
          "total_time": videonun toplam süresi (saniye),
          "total_active_time": tespit edilen toplam aktif süre (saniye),
          "active_segments": [her bir aktif segmentin uzunluğu (saniye) listesi],
          "mean_segment_duration": aktif segment ortalama süresi,
          "std_segment_duration": aktif segment sürelerinin standart sapması
        }
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video açılamadı: {video_path}")
    
    # Videonun FPS ve toplam kare sayısı bilgisi
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps  # Videonun toplam süresi (saniye)

    # Hareket algılayıcı (background subtractor)
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )

    active_segments = []
    segment_start = None
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = back_sub.apply(frame)

        # 127 üstü pikselleri hareket olarak kabul edelim
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        motion_area = cv2.countNonZero(thresh)

        if motion_area > min_area_threshold:  # kare aktif
            if segment_start is None:
                segment_start = frame_index
        else:  # kare pasif
            if segment_start is not None:
                segment_end = frame_index
                duration_in_seconds = (segment_end - segment_start) / fps
                active_segments.append(duration_in_seconds)
                segment_start = None

        frame_index += 1

    # Döngü bitti, hâlâ açık segment varsa
    if segment_start is not None:
        segment_end = frame_index
        duration_in_seconds = (segment_end - segment_start) / fps
        active_segments.append(duration_in_seconds)

    cap.release()

    total_active_time = sum(active_segments)

    if len(active_segments) > 0:
        mean_segment_duration = statistics.mean(active_segments)
        std_segment_duration = (
            statistics.stdev(active_segments) if len(active_segments) > 1 else 0.0
        )
    else:
        mean_segment_duration = 0.0
        std_segment_duration = 0.0

    return {
        "fps": fps,
        "frame_count": total_frames,
        "total_time": total_duration,
        "total_active_time": total_active_time,
        "active_segments": active_segments,
        "mean_segment_duration": mean_segment_duration,
        "std_segment_duration": std_segment_duration,
    }

def main():
    st.title("Video Aktif Süre Analizi (Hareket Algılama)")

    uploaded_video = st.file_uploader(
        "Bir video dosyası yükleyin", 
        type=["mp4", "avi", "mov", "mkv"]
    )
    
    min_area_threshold = st.slider(
        "Hareket Eşiği (piksel)", 
        min_value=1000, 
        max_value=20000, 
        value=5000, 
        step=500
    )
    
    if uploaded_video is not None:
        st.video(uploaded_video)

        # Geçici dosya
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_video.read())
            temp_path = temp_file.name

        with st.spinner("Analiz yapılıyor..."):
            result = analyze_video_worker_time(
                temp_path, 
                min_area_threshold=min_area_threshold
            )
        
        st.subheader("Analiz Sonuçları")
        st.write(f"**FPS**: {result['fps']}")
        st.write(f"**Toplam Kare**: {result['frame_count']}")
        st.write(f"**Toplam Video Süresi**: {result['total_time']:.2f} sn")
        st.write(f"**Toplam Aktif Süre**: {result['total_active_time']:.2f} sn")
        st.write(f"**Aktif Segmentler (saniye)**: {result['active_segments']}")
        st.write(f"**Ortalama Segment Süresi**: {result['mean_segment_duration']:.2f} sn")
        st.write(f"**Segment Süreleri Standart Sapması**: {result['std_segment_duration']:.2f} sn")

        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()
