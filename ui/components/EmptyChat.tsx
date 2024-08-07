import EmptyChatMessageInput from './EmptyChatMessageInput';
import image from '../public/apple-touch-icon.png';

const EmptyChat = ({
  sendMessage,
  focusMode,
  setFocusMode,
}: {
  sendMessage: (message: string) => void;
  focusMode: string;
  setFocusMode: (mode: string) => void;
}) => {
  return (
    <div className="relative">
      <div className="flex flex-col items-center justify-center min-h-screen max-w-screen-sm mx-auto p-2 space-y-8">
        
      <header className="flex flex-col items-center p-4 space-y-2">
          <img src={image.src} alt="Company Logo" className="w-24 h-20" />
          <h1 className="text-2xl lg:text-3xl font-bold" style={{ color: '#fd7217' }}>Konect U</h1>
        </header>

        <h2 className="text-black/70 dark:text-white/70 text-3xl font-medium -mt-8">
           HEY THERE, I AM NALANDA
        </h2>
        <EmptyChatMessageInput
          sendMessage={sendMessage}
          focusMode={focusMode}
          setFocusMode={setFocusMode}
        />
      </div>
    </div>
  );
};

export default EmptyChat;
