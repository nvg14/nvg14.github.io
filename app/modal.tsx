import React, { useState } from 'react';
import Image from 'next/image';
import {
  Box,
  Typography,
  Modal,
  Fade,
  Backdrop,
  ToggleButton,
  ToggleButtonGroup,
  Slider,
  Button,
  TextField,
  MenuItem
} from '@mui/material';

const SizeRecommendationModal = ({
  open,
  onClose,
  onSelectSize,
}: {
  open: boolean;
  onClose: () => void;
  onSelectSize: (size: string) => void;
}) => {
  const [fit, setFit] = useState('regular');
  const [height, setHeight] = useState(165);
  const [weight, setWeight] = useState(60);
  const [bodyShape, setBodyShape] = useState('');
  const [brand, setBrand] = useState('Zara');
  const [lastSize, setLastSize] = useState('M');
  const [recommendedSize, setRecommendedSize] = useState('');
  const [fitNote, setFitNote] = useState('');

  const bodyShapeOptions = ['🍎 Apple', '🍐 Pear', '⏳ Hourglass', '🏋️ Athletic'];
  const brandOptions = ['Zara', 'H&M', 'Levi’s'];
  const sizeOptions = ['XS', 'S', 'M', 'L', 'XL'];

  const handleRecommend = () => {
    let size = lastSize;

    // Simple logic
    if (weight > 80 || height > 180) size = 'L';
    else if (weight < 50 || height < 155) size = 'S';
    else size = 'M';

    // Adjust for body shape
    if (bodyShape === '⏳ Hourglass' && fit === 'tight') size = 'L';
    if (bodyShape === '🏋️ Athletic' && fit === 'tight') size = 'L';

    // Fit notes
    let note = '';
    if (fit === 'tight') note = 'Slim Fit – May be snug on shoulders';
    else if (fit === 'loose') note = 'Loose Fit – Comfort guaranteed';
    else note = 'Standard Fit – Balanced cut';

    setRecommendedSize(size);
    setFitNote(note);
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      closeAfterTransition
      BackdropComponent={Backdrop}
      BackdropProps={{ timeout: 300 }}
    >
      <Fade in={open}>
        <Box
          sx={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: 500,
            bgcolor: "background.paper",
            borderRadius: 2,
            boxShadow: 24,
            p: 4,
          }}
        >
          <Typography variant="h6" mb={2} fontWeight={600}>
            Size Selector
          </Typography>

          {/* Fit Preference */}
          <Typography variant="subtitle1" fontWeight={500}>
            Preferred fit
          </Typography>
          <ToggleButtonGroup
            color="primary"
            exclusive
            value={fit}
            onChange={(e, newFit) => newFit && setFit(newFit)}
            sx={{ mt: 1, mb: 2 }}
          >
            <ToggleButton value="tight">Tight</ToggleButton>
            <ToggleButton value="regular">Regular</ToggleButton>
            <ToggleButton value="loose">Loose</ToggleButton>
          </ToggleButtonGroup>

          {/* Height & Weight */}
          <Typography variant="subtitle1" fontWeight={500}>
            Height (cm): {height} cm
          </Typography>
        
          <Slider
            value={height}
            min={140}
            max={200}
            onChange={(e, val) => {
              if (typeof val === 'number') {
                setHeight(val);
              }
            }}
            sx={{ mt: 1, mb: 2 }}
          />
            
          <Typography variant="subtitle1" fontWeight={500}>
            Weight (kg): {weight} kg
          </Typography>
          <Slider
            value={weight}
            min={35}
            max={120}
            onChange={(e, val) => {
              if (typeof val === 'number') {
                setWeight(val);
              }
            }}
            sx={{ mt: 1, mb: 2 }}
          />

          {/* Body Shape */}
          <Typography variant="subtitle1" fontWeight={500} mb={1}>
            Select body shape image
          </Typography>
          <Box display="flex" gap={1.5} mb={2}>
            {bodyShapeOptions.map((shape) => (
              <Button
                key={shape}
                variant={bodyShape === shape ? "contained" : "outlined"}
                onClick={() => setBodyShape(shape)}
              >
                {shape}
              </Button>
            ))}
          </Box>

          {/* Brand & Size */}
          <Typography variant="subtitle1" fontWeight={500} mb={1}>
            Recent size in known brands
          </Typography>
          <Box display="flex" gap={2} mb={3}>
            <TextField
              label="Brand"
              select
              fullWidth
              size="small"
              value={brand}
              onChange={(e) => setBrand(e.target.value)}
            >
              {brandOptions.map((b) => (
                <MenuItem key={b} value={b}>
                  {b}
                </MenuItem>
              ))}
            </TextField>
            <TextField
              label="Size"
              select
              fullWidth
              size="small"
              value={lastSize}
              onChange={(e) => setLastSize(e.target.value)}
            >
              {sizeOptions.map((s) => (
                <MenuItem key={s} value={s}>
                  {s}
                </MenuItem>
              ))}
            </TextField>
          </Box>

          {/* Recommendation */}
          {recommendedSize && (
            <Box
            display="flex"
            flexDirection="column" // 👈 key change
            gap={2}
            alignItems="flex-start"
            p={2}
            sx={{ bgcolor: "#f7f7f7", borderRadius: 2, mb: 3 }}
          >
            <Box
            display="flex"
            flexDirection="row" // 👈 key change
            gap={2}
            alignItems="flex-start"
            p={2}
            sx={{ bgcolor: "#f7f7f7", borderRadius: 2, mb: 3 }}
          >
            <Box sx={{ fontSize: 40 }}>🧍</Box>
          
            <Box>
              <Typography variant="subtitle1" fontWeight={600}>
                Recommendation
              </Typography>
              <Typography>
                Size: <strong>{recommendedSize}</strong>
              </Typography>
              <Typography>
                Fit: <strong>{fitNote}</strong>
              </Typography>
            </Box>
          </Box>
          <div
              style={{
                position: 'relative',
                width: '100%',
                height: '150px',
                overflow: 'hidden',
              }}
            >
              <Image
                src="/size.png"
                alt="Background Image"
                layout="fill"
                quality={100}
                priority
              />
            </div>
          </Box>
          
          )}

          {/* <Button
            variant="contained"
            fullWidth
            onClick={() => {
              handleRecommend();
            }}
          >
            {recommendedSize ? "Accept Recommendation" : "Get Recommendation"}
          </Button> */}

          <Button
            variant="contained"
            fullWidth
            onClick={() => {
              if (recommendedSize) {
                onSelectSize(recommendedSize); // Pass selected size to page.tsx
                onClose();                     // Close the modal
              } else {
                handleRecommend();            // Trigger recommendation logic
              }
            }}
          >
            {recommendedSize ? "Accept Recommendation" : "Get Recommendation"}
          </Button>
        </Box>
      </Fade>
    </Modal>
  );
};

export default SizeRecommendationModal;
