import { useCallback, useMemo, useState } from 'react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const ROLE_OPTIONS = [
  { value: 'driver', label: 'Driver (lowest weight)' },
  { value: 'manager', label: 'Manager (medium weight)' },
  { value: 'owner', label: 'Owner (highest weight)' },
];


function Message({ message }) {
  return (
    <div className={`message message-${message.role}`}>
      <div className="message-role">{message.role === 'user' ? 'You' : 'Assistant'}</div>
      <div className="message-content">{message.content}</div>
      {message.documents?.length ? (
        <details className="context-details">
          <summary>Referenced documents ({message.documents.length})</summary>
          <ul>
            {message.documents.map((doc, index) => (
              <li key={`${doc.source}-${index}`}>
                <strong>{doc.source || `Chunk ${index + 1}`}</strong>
                <div>{doc.content.slice(0, 300)}{doc.content.length > 300 ? '…' : ''}</div>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {message.feedback?.length ? (
        <details className="context-details">
          <summary>Applied user feedback ({message.feedback.length})</summary>
          <ul>
            {message.feedback.map((item) => (
              <li key={item.id}>
                <strong>{item.user_role}</strong> · weight {item.weight} · score {item.score.toFixed(2)}
                <div className="feedback-response">{item.updated_response}</div>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

function UploadPanel({ onUpload }) {
  const [status, setStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  const handleChange = async (event) => {
    const files = event.target.files;
    if (!files?.length) {
      return;
    }
    setIsUploading(true);
    setStatus('Uploading...');
    try {
      await onUpload(files);
      setStatus(`Uploaded ${files.length} file${files.length > 1 ? 's' : ''} successfully.`);
    } catch (error) {
      console.error(error);
      setStatus(error?.response?.data?.detail || 'Upload failed.');
    } finally {
      setIsUploading(false);
      event.target.value = '';
    }
  };

  return (
    <div className="upload-panel">
      <h2>Document upload</h2>
      <p>Upload knowledge base files (.txt, .md, .json, .pdf) to feed the retriever.</p>
      <label className={`upload-button ${isUploading ? 'disabled' : ''}`}>
        <span>{isUploading ? 'Uploading…' : 'Select files'}</span>
        <input type="file" multiple onChange={handleChange} disabled={isUploading} hidden />
      </label>
      {status ? <p className="status-text">{status}</p> : null}
    </div>
  );
}

export default function App() {
  const [role, setRole] = useState('driver');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [pendingFeedback, setPendingFeedback] = useState(null);
  const [feedbackText, setFeedbackText] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState('');

  const api = useMemo(
    () =>
      axios.create({
        baseURL: API_BASE_URL,
      }),
    [],
  );

  const handleUpload = useCallback(
    async (fileList) => {
      const formData = new FormData();
      Array.from(fileList).forEach((file) => formData.append('files', file));
      await api.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    },
    [api],
  );

  const sendMessage = useCallback(async () => {
    if (!input.trim()) {
      return;
    }

    const query = input.trim();
    const history = messages.map(({ role: messageRole, content }) => ({ role: messageRole, content }));
    const optimisticMessages = [...messages, { role: 'user', content: query }];
    setMessages(optimisticMessages);
    setInput('');
    setIsSending(true);
    setFeedbackStatus('');

    try {
      const { data } = await api.post('/chat', {
        query,
        user_role: role,
        chat_history: history,
      });
      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        documents: data.used_documents || [],
        feedback: data.applied_feedback || [],
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setPendingFeedback({
        query,
        response: data.answer,
        user_role: role,
      });
      setFeedbackText(data.answer);
    } catch (error) {
      console.error(error);
      const errorMessage = error?.response?.data?.detail || 'Unable to fetch assistant response.';
      setMessages((prev) => [...prev, { role: 'assistant', content: errorMessage }]);
    } finally {
      setIsSending(false);
    }
  }, [api, input, messages, role]);

  const submitFeedback = useCallback(async () => {
    if (!pendingFeedback) {
      setFeedbackStatus('Ask a question and receive a response before submitting feedback.');
      return;
    }
    if (!feedbackText.trim()) {
      setFeedbackStatus('Provide an updated response before submitting feedback.');
      return;
    }

    try {
      const payload = {
        ...pendingFeedback,
        updated_response: feedbackText.trim(),
      };
      const { data } = await api.post('/feedback', payload);
      setFeedbackStatus(`Feedback saved for role ${data.entry.user_role}.`);
      setPendingFeedback(null);
    } catch (error) {
      console.error(error);
      setFeedbackStatus(error?.response?.data?.detail || 'Failed to save feedback.');
    }
  }, [api, feedbackText, pendingFeedback]);

  return (
    <div className="app">
      <header>
        <h1>RAG Feedback Assistant</h1>
        <p>Search company knowledge, adjust the answer, and loop improvements back into retrieval.</p>
      </header>

      <section className="controls">
        <div>
          <label htmlFor="role-select">Your role:</label>
          <select
            id="role-select"
            value={role}
            onChange={(event) => setRole(event.target.value)}
          >
            {ROLE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </section>

      <UploadPanel onUpload={handleUpload} />

      <section className="chat">
        <div className="messages">
          {messages.length === 0 ? <p className="empty-state">Upload documents and start asking questions.</p> : null}
          {messages.map((message, index) => (
            <Message key={`${message.role}-${index}`} message={message} />
          ))}
        </div>
        <div className="composer">
          <textarea
            placeholder="Ask a question about your documents…"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            rows={3}
          />
          <button type="button" onClick={sendMessage} disabled={isSending}>
            {isSending ? 'Thinking…' : 'Send'}
          </button>
        </div>
      </section>

      <section className="feedback-editor">
        <h2>Review and update the assistant response</h2>
        <p>Adjust the assistant answer to reflect real-world processes. Higher-role feedback outweighs lower-role feedback during future retrieval.</p>
        <textarea
          value={feedbackText}
          onChange={(event) => setFeedbackText(event.target.value)}
          placeholder="Edit the assistant response here before submitting."
          rows={6}
        />
        <button type="button" onClick={submitFeedback}>Submit feedback</button>
        {feedbackStatus ? <p className="status-text">{feedbackStatus}</p> : null}
      </section>
    </div>
  );
}
