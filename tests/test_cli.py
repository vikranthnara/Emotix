"""Unit tests for interactive CLI."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, call
import sys

from src.persistence import MWBPersistence
from src.preprocess import Preprocessor
from src.modeling import EmotionModel
from src.postprocess import EmotionPostProcessor
from src.suicidal_detection import SuicidalIdeationDetector


# Import CLI functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from emotix_cli import process_single_entry, print_result, print_welcome


class TestCLIFunctions:
    """Test CLI utility functions."""
    
    def test_print_welcome(self, capsys):
        """Test welcome message printing."""
        print_welcome()
        captured = capsys.readouterr()
        assert "Emotix" in captured.out
        assert "Interactive CLI" in captured.out
        assert "Commands:" in captured.out
        assert "exit" in captured.out
        assert "summary" in captured.out
    
    def test_print_result(self, capsys):
        """Test result printing."""
        result = {
            'text': 'test entry',
            'normalized_text': 'test entry normalized',
            'emotion': 'joy',
            'intensity': 0.85,
            'original_emotion': 'joy',
            'was_overridden': False,
            'override_type': None
        }
        print_result(result)
        captured = capsys.readouterr()
        assert "test entry" in captured.out
        assert "joy" in captured.out
        assert "0.85" in captured.out
    
    def test_print_result_with_override(self, capsys):
        """Test result printing with override."""
        result = {
            'text': 'thx for help',
            'normalized_text': 'thanks for help',
            'emotion': 'joy',
            'intensity': 0.90,
            'original_emotion': 'sadness',
            'was_overridden': True,
            'override_type': 'positive_keywords'
        }
        print_result(result)
        captured = capsys.readouterr()
        assert "Override" in captured.out
        assert "sadness" in captured.out and "joy" in captured.out
        assert "positive_keywords" in captured.out


class TestProcessSingleEntry:
    """Test single entry processing."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock emotion model."""
        model = Mock(spec=EmotionModel)
        return model
    
    @pytest.fixture
    def mock_postprocessor(self):
        """Create mock post-processor."""
        postprocessor = Mock(spec=EmotionPostProcessor)
        postprocessor.post_process.return_value = ('joy', 0.85, False, None)
        return postprocessor
    
    @pytest.fixture
    def mock_suicidal_detector(self):
        """Create mock suicidal ideation detector."""
        detector = Mock(spec=SuicidalIdeationDetector)
        detector.process_text.return_value = (False, 0.0, None)
        return detector
    
    def test_process_single_entry_basic(self, temp_db, mock_model, mock_postprocessor, mock_suicidal_detector):
        """Test basic single entry processing."""
        persistence = MWBPersistence(temp_db)
        preprocessor = Preprocessor()
        
        # Mock run_inference_pipeline to return proper DataFrame
        with patch('emotix_cli.run_inference_pipeline') as mock_inference:
            mock_inference.return_value = pd.DataFrame({
                'UserID': ['testuser'],
                'Text': ['I am happy today'],
                'Timestamp': [datetime(2024, 1, 1, 10, 0)],
                'NormalizedText': ['I am happy today'],
                'NormalizationFlags': [{}],
                'Sequence': ['I am happy today'],
                'PrimaryEmotionLabel': ['joy'],
                'IntensityScore_Primary': [0.85],
                'AmbiguityFlag': [False]
            })
            
            result = process_single_entry(
                text="I am happy today",
                user_id="testuser",
                timestamp=datetime(2024, 1, 1, 10, 0),
                persistence=persistence,
                preprocessor=preprocessor,
                model=mock_model,
                postprocessor=mock_postprocessor,
                suicidal_detector=mock_suicidal_detector,
                slang_dict_path=None
            )
        
        assert result['text'] == "I am happy today"
        assert 'normalized_text' in result
        assert 'emotion' in result
        assert 'intensity' in result
        
        # Verify entry was persisted
        history = persistence.fetch_history("testuser")
        assert len(history) == 1
        assert "happy" in history['NormalizedText'].iloc[0].lower()
    
    def test_process_single_entry_with_context(self, temp_db, mock_model, mock_postprocessor, mock_suicidal_detector):
        """Test single entry processing with existing history."""
        persistence = MWBPersistence(temp_db)
        preprocessor = Preprocessor()
        
        # Add previous entry
        df_prev = pd.DataFrame({
            'UserID': ['testuser'],
            'Timestamp': [datetime(2024, 1, 1, 9, 0)],
            'Text': ['Previous entry'],
            'NormalizedText': ['Previous entry'],
            'NormalizationFlags': [{}],
            'PrimaryEmotionLabel': ['joy'],
            'IntensityScore_Primary': [0.8]
        })
        persistence.write_results(df_prev, archive_raw=False)
        
        # Mock run_inference_pipeline
        with patch('emotix_cli.run_inference_pipeline') as mock_inference:
            mock_inference.return_value = pd.DataFrame({
                'UserID': ['testuser'],
                'Text': ['Another entry'],
                'Timestamp': [datetime(2024, 1, 1, 10, 0)],
                'NormalizedText': ['Another entry'],
                'NormalizationFlags': [{}],
                'Sequence': ['Another entry [SEP] Previous entry'],
                'PrimaryEmotionLabel': ['sadness'],
                'IntensityScore_Primary': [0.75],
                'AmbiguityFlag': [False]
            })
            
            # Process new entry
            result = process_single_entry(
                text="Another entry",
                user_id="testuser",
                timestamp=datetime(2024, 1, 1, 10, 0),
                persistence=persistence,
                preprocessor=preprocessor,
                model=mock_model,
                postprocessor=mock_postprocessor,
                suicidal_detector=mock_suicidal_detector,
                slang_dict_path=None
            )
        
        # Verify both entries exist
        history = persistence.fetch_history("testuser")
        assert len(history) == 2


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""
    
    def test_parse_arguments_default(self):
        """Test default argument parsing."""
        from emotix_cli import main
        import argparse
        
        # Test with no arguments (will fail at user input, but we can test parsing)
        with patch('emotix_cli.interactive_loop') as mock_loop:
            with patch('builtins.input', return_value='testuser'):
                with patch('sys.argv', ['emotix_cli.py']):
                    try:
                        main()
                    except SystemExit:
                        pass  # argparse may call sys.exit
    
    def test_parse_arguments_with_user(self):
        """Test argument parsing with --user flag."""
        from emotix_cli import main
        
        with patch('emotix_cli.interactive_loop') as mock_loop:
            with patch('sys.argv', ['emotix_cli.py', '--user', 'testuser']):
                try:
                    main()
                except SystemExit:
                    pass  # argparse may call sys.exit
    
    def test_parse_arguments_with_db(self):
        """Test argument parsing with --db flag."""
        from emotix_cli import main
        
        with patch('emotix_cli.interactive_loop') as mock_loop:
            with patch('sys.argv', ['emotix_cli.py', '--user', 'testuser', '--db', 'custom.db']):
                try:
                    main()
                except SystemExit:
                    pass


class TestCLIInteractiveLoop:
    """Test interactive loop functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock emotion model."""
        model = Mock(spec=EmotionModel)
        return model
    
    @pytest.fixture
    def mock_postprocessor(self):
        """Create mock post-processor."""
        postprocessor = Mock(spec=EmotionPostProcessor)
        postprocessor.post_process.return_value = ('joy', 0.85, False, None)
        return postprocessor
    
    def test_interactive_loop_exit_command(self, temp_db, mock_model, mock_postprocessor):
        """Test exit command in interactive loop."""
        from emotix_cli import interactive_loop
        
        persistence = MWBPersistence(temp_db)
        preprocessor = Preprocessor()
        
        # Mock input to return 'exit' immediately
        with patch('builtins.input', side_effect=['exit']):
            with patch('emotix_cli.process_single_entry') as mock_process:
                with patch('emotix_cli.EmotionModel') as mock_model_class:
                    with patch('emotix_cli.SuicidalIdeationDetector') as mock_detector_class:
                        mock_model_class.return_value = mock_model
                        mock_detector = Mock()
                        mock_detector_class.return_value = mock_detector
                        interactive_loop(
                            user_id="testuser",
                            db_path=Path(temp_db),
                            slang_dict_path=None,
                            model_name="j-hartmann/emotion-english-distilroberta-base"
                        )
                        # Should not call process_single_entry for exit command
                        mock_process.assert_not_called()
    
    def test_interactive_loop_quit_command(self, temp_db, mock_model, mock_postprocessor):
        """Test quit command in interactive loop."""
        from emotix_cli import interactive_loop
        
        persistence = MWBPersistence(temp_db)
        preprocessor = Preprocessor()
        
        # Mock input to return 'quit' immediately
        with patch('builtins.input', side_effect=['quit']):
            with patch('emotix_cli.process_single_entry') as mock_process:
                with patch('emotix_cli.EmotionModel') as mock_model_class:
                    with patch('emotix_cli.SuicidalIdeationDetector') as mock_detector_class:
                        mock_model_class.return_value = mock_model
                        mock_detector = Mock()
                        mock_detector_class.return_value = mock_detector
                        interactive_loop(
                            user_id="testuser",
                            db_path=Path(temp_db),
                            slang_dict_path=None,
                            model_name="j-hartmann/emotion-english-distilroberta-base"
                        )
                        mock_process.assert_not_called()
    
    def test_interactive_loop_summary_command(self, temp_db, capsys):
        """Test summary command in interactive loop."""
        from emotix_cli import interactive_loop
        
        # Create persistence with some test data and a journal
        persistence = MWBPersistence(temp_db)
        journal_id = persistence.create_journal("testuser", "Test Journal")
        
        df = pd.DataFrame({
            'UserID': ['testuser', 'testuser'],
            'Timestamp': [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0)],
            'Text': ['Entry 1', 'Entry 2'],
            'NormalizedText': ['Entry 1', 'Entry 2'],
            'NormalizationFlags': [{}, {}],
            'PrimaryEmotionLabel': ['joy', 'sadness'],
            'IntensityScore_Primary': [0.8, 0.9]
        })
        persistence.write_results(df, archive_raw=False, journal_id=journal_id)
        
        # Mock model and postprocessor
        mock_model = Mock(spec=EmotionModel)
        mock_postprocessor = Mock(spec=EmotionPostProcessor)
        
        # Mock input to select journal (1), then 'summary', then 'exit'
        with patch('builtins.input', side_effect=['1', 'summary', 'exit']):
            with patch('emotix_cli.EmotionModel', return_value=mock_model):
                with patch('emotix_cli.EmotionPostProcessor', return_value=mock_postprocessor):
                    with patch('emotix_cli.SuicidalIdeationDetector') as mock_detector_class:
                        with patch('src.preprocess.Preprocessor') as mock_preprocessor_class:
                            mock_detector = Mock()
                            mock_detector.process_text.return_value = (False, 0.0, None)
                            mock_detector_class.return_value = mock_detector
                            mock_preprocessor = Mock()
                            mock_preprocessor.preprocess_dataframe.return_value = df
                            mock_preprocessor_class.return_value = mock_preprocessor
                            interactive_loop(
                                user_id="testuser",
                                db_path=Path(temp_db),
                                slang_dict_path=None,
                                model_name="j-hartmann/emotion-english-distilroberta-base"
                            )
        
        captured = capsys.readouterr()
        assert "Last 3 Entries Summary" in captured.out or "Last 3 Entries" in captured.out
    
    def test_interactive_loop_history_command(self, temp_db, capsys):
        """Test history command in interactive loop."""
        from emotix_cli import interactive_loop
        
        # Create persistence with test data and a journal
        persistence = MWBPersistence(temp_db)
        journal_id = persistence.create_journal("testuser", "Test Journal")
        
        df = pd.DataFrame({
            'UserID': ['testuser'] * 3,
            'Timestamp': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 1, 12, 0)
            ],
            'Text': ['Entry 1', 'Entry 2', 'Entry 3'],
            'NormalizedText': ['Entry 1', 'Entry 2', 'Entry 3'],
            'NormalizationFlags': [{}, {}, {}],
            'PrimaryEmotionLabel': ['joy', 'sadness', 'anger'],
            'IntensityScore_Primary': [0.8, 0.9, 0.7]
        })
        persistence.write_results(df, archive_raw=False, journal_id=journal_id)
        
        # Mock model and postprocessor
        mock_model = Mock(spec=EmotionModel)
        mock_postprocessor = Mock(spec=EmotionPostProcessor)
        
        # Mock input to select journal (1), then 'history', then 'exit'
        with patch('builtins.input', side_effect=['1', 'history', 'exit']):
            with patch('emotix_cli.EmotionModel', return_value=mock_model):
                with patch('emotix_cli.EmotionPostProcessor', return_value=mock_postprocessor):
                    with patch('emotix_cli.SuicidalIdeationDetector') as mock_detector_class:
                        with patch('src.preprocess.Preprocessor') as mock_preprocessor_class:
                            mock_detector = Mock()
                            mock_detector.process_text.return_value = (False, 0.0, None)
                            mock_detector_class.return_value = mock_detector
                            mock_preprocessor = Mock()
                            mock_preprocessor.preprocess_dataframe.return_value = df
                            mock_preprocessor_class.return_value = mock_preprocessor
                            interactive_loop(
                                user_id="testuser",
                                db_path=Path(temp_db),
                                slang_dict_path=None,
                                model_name="j-hartmann/emotion-english-distilroberta-base"
                            )
        
        captured = capsys.readouterr()
        assert "History" in captured.out or "history" in captured.out.lower()
    
    def test_interactive_loop_help_command(self, capsys):
        """Test help command in interactive loop."""
        from emotix_cli import interactive_loop
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create a journal first
            persistence = MWBPersistence(temp_path)
            persistence.create_journal("testuser", "Test Journal")
            
            # Mock input to select journal (1), then 'help', then 'exit'
            with patch('builtins.input', side_effect=['1', 'help', 'exit']):
                with patch('emotix_cli.EmotionModel') as mock_model_class:
                    with patch('emotix_cli.EmotionPostProcessor') as mock_postprocessor_class:
                        with patch('emotix_cli.SuicidalIdeationDetector') as mock_detector_class:
                            mock_model_class.return_value = Mock(spec=EmotionModel)
                            mock_postprocessor_class.return_value = Mock(spec=EmotionPostProcessor)
                            mock_detector = Mock()
                            mock_detector.process_text.return_value = (False, 0.0, None)
                            mock_detector_class.return_value = mock_detector
                            with patch('src.preprocess.Preprocessor') as mock_preprocessor_class:
                                mock_preprocessor = Mock()
                                mock_preprocessor_class.return_value = mock_preprocessor
                                interactive_loop(
                                    user_id="testuser",
                                    db_path=Path(temp_path),
                                    slang_dict_path=None,
                                    model_name="j-hartmann/emotion-english-distilroberta-base"
                                )
            
            captured = capsys.readouterr()
            # Help should show welcome message
            assert "Emotix" in captured.out or "Commands" in captured.out
        finally:
            Path(temp_path).unlink()
    
    def test_interactive_loop_process_entry(self, temp_db, capsys):
        """Test processing an actual entry in interactive loop."""
        from emotix_cli import interactive_loop
        
        # Create a journal first
        persistence = MWBPersistence(temp_db)
        persistence.create_journal("testuser", "Test Journal")
        
        # Mock model to avoid loading actual transformer
        mock_model = Mock(spec=EmotionModel)
        mock_model.predict_batch.return_value = pd.DataFrame({
            'PrimaryEmotionLabel': ['joy'],
            'IntensityScore_Primary': [0.85],
            'AmbiguityFlag': [False]
        })
        
        mock_postprocessor = Mock(spec=EmotionPostProcessor)
        mock_postprocessor.post_process.return_value = ('joy', 0.85, False, None)
        
        # Mock input to select journal (1), process an entry, then exit
        with patch('builtins.input', side_effect=['1', 'I am happy', 'exit']):
            with patch('emotix_cli.EmotionModel', return_value=mock_model):
                with patch('emotix_cli.EmotionPostProcessor', return_value=mock_postprocessor):
                    with patch('emotix_cli.SuicidalIdeationDetector') as mock_detector_class:
                        with patch('emotix_cli.run_inference_pipeline') as mock_inference:
                            mock_detector = Mock()
                            mock_detector.process_text.return_value = (False, 0.0, None)
                            mock_detector_class.return_value = mock_detector
                            
                            # Mock inference to return DataFrame with predictions
                            mock_inference.return_value = pd.DataFrame({
                                'UserID': ['testuser'],
                                'Text': ['I am happy'],
                                'Timestamp': [datetime(2024, 1, 1, 10, 0)],
                                'NormalizedText': ['I am happy'],
                                'NormalizationFlags': [{}],
                                'Sequence': ['I am happy'],
                                'PrimaryEmotionLabel': ['joy'],
                                'IntensityScore_Primary': [0.85],
                                'AmbiguityFlag': [False]
                            })
                            
                            interactive_loop(
                                user_id="testuser",
                                db_path=Path(temp_db),
                                slang_dict_path=None,
                                model_name="j-hartmann/emotion-english-distilroberta-base"
                            )
        
        captured = capsys.readouterr()
        assert "Processed" in captured.out or "happy" in captured.out.lower()
    
    def test_interactive_loop_keyboard_interrupt(self, temp_db):
        """Test handling KeyboardInterrupt in interactive loop."""
        from emotix_cli import interactive_loop
        
        # Create a journal first
        persistence = MWBPersistence(temp_db)
        persistence.create_journal("testuser", "Test Journal")
        
        mock_model = Mock(spec=EmotionModel)
        mock_postprocessor = Mock(spec=EmotionPostProcessor)
        
        # Mock input to select journal (1), then raise KeyboardInterrupt
        with patch('builtins.input', side_effect=['1', KeyboardInterrupt()]):
            with patch('emotix_cli.EmotionModel', return_value=mock_model):
                with patch('emotix_cli.EmotionPostProcessor', return_value=mock_postprocessor):
                    with patch('emotix_cli.SuicidalIdeationDetector') as mock_detector_class:
                        with patch('src.preprocess.Preprocessor') as mock_preprocessor_class:
                            mock_detector = Mock()
                            mock_detector.process_text.return_value = (False, 0.0, None)
                            mock_detector_class.return_value = mock_detector
                            mock_preprocessor = Mock()
                            mock_preprocessor_class.return_value = mock_preprocessor
                            # Should handle KeyboardInterrupt gracefully
                            try:
                                interactive_loop(
                                    user_id="testuser",
                                    db_path=Path(temp_db),
                                    slang_dict_path=None,
                                    model_name="j-hartmann/emotion-english-distilroberta-base"
                                )
                            except KeyboardInterrupt:
                                pytest.fail("KeyboardInterrupt should be handled gracefully")
    
    def test_interactive_loop_empty_input(self, temp_db):
        """Test handling empty input in interactive loop."""
        from emotix_cli import interactive_loop
        
        # Create a journal first
        persistence = MWBPersistence(temp_db)
        persistence.create_journal("testuser", "Test Journal")
        
        mock_model = Mock(spec=EmotionModel)
        mock_postprocessor = Mock(spec=EmotionPostProcessor)
        
        # Mock input to select journal (1), return empty string, then exit
        with patch('builtins.input', side_effect=['1', '', '   ', 'exit']):
            with patch('emotix_cli.EmotionModel', return_value=mock_model):
                with patch('emotix_cli.EmotionPostProcessor', return_value=mock_postprocessor):
                    with patch('emotix_cli.SuicidalIdeationDetector') as mock_detector_class:
                        with patch('src.preprocess.Preprocessor') as mock_preprocessor_class:
                            mock_detector = Mock()
                            mock_detector.process_text.return_value = (False, 0.0, None)
                            mock_detector_class.return_value = mock_detector
                            mock_preprocessor = Mock()
                            mock_preprocessor_class.return_value = mock_preprocessor
                            with patch('emotix_cli.process_single_entry') as mock_process:
                                interactive_loop(
                                    user_id="testuser",
                                    db_path=Path(temp_db),
                                    slang_dict_path=None,
                                    model_name="j-hartmann/emotion-english-distilroberta-base"
                                )
                            # Empty inputs should not trigger processing
                            mock_process.assert_not_called()

