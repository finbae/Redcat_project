# Redcat_project
### opencv를 이용한 손 인식
    import cv2
    import mediapipe as mp

    # MediaPipe Hand Tracking 초기화
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    def track_hand_motion():
      # 웹캠 열기    
      cap = cv2.VideoCapture(0)
      # MediaPipe Hand Tracking 인스턴스 생성
      with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
          ret, frame = cap.read()
      
        if not ret:
          print("비디오 읽기 실패")
          break
      
        # BGR 이미지를 RGB 이미지로 변환           
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # 이미지에서 손 추적            
        results = hands.process(image)
      
        # RGB 이미지를 다시 BGR 이미지로 변환            
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
        # 손 추적 결과를 이용하여 손 모션 디스플레이            
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
          
        # 화면에 출력            
        cv2.imshow('Hand Motion', image)
      
        # 'q' 키를 누르면 종료            
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break 
      
      # 자원 해제    
      cap.release()
      cv2.destroyAllWindows()
    
    # 손 모션 추적 시작
    track_hand_motion()
