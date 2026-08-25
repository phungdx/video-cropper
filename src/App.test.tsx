import { render, screen } from '@testing-library/react';
import App from './App';

describe('App', () => {
  it('shows the video upload entry point', () => {
    render(<App />);

    expect(screen.getByText(/select a person and let the crop follow them/i)).toBeInTheDocument();
    expect(screen.getByText(/choose a video/i)).toBeInTheDocument();
  });
});
